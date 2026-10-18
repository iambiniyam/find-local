"""Command line interface."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import db_path
from .embed import build_embedder
from .index import Indexer, IndexMismatch, Store
from .search import search as run_search
from .watch import watch as run_watch

app = typer.Typer(add_completion=False, help="Search your computer by describing it, offline.")
console = Console(width=130)


@app.callback(invoke_without_command=True)
def _root(
    version: bool = typer.Option(False, "--version", is_eager=True, help="Show the version and exit."),
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


def _store(path: Path) -> Store:
    return Store(path)


@app.command("index")
def index_command(
    paths: list[Path] = typer.Argument(..., exists=True, help="Files or folders to index."),
    db: Path = typer.Option(db_path(), help="SQLite index path."),
    embedder: str = typer.Option("auto", help="auto or hash (tests)."),
    no_images: bool = typer.Option(False, "--no-images", help="Skip image embeddings."),
    ocr: bool = typer.Option(False, "--ocr", help="Extract text from images with RapidOCR."),
) -> None:
    """Index files, PDFs, and images."""
    store = _store(db)
    indexer = Indexer(store, build_embedder(embedder, with_images=not no_images), ocr=ocr)
    try:
        stats = indexer.index_paths(paths)
    except IndexMismatch as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"Delete the index and run again:\n  rm {db}")
        raise typer.Exit(code=2) from exc
    console.print(
        f"[green]indexed[/green] {stats.files} files · {stats.chunks} chunks · "
        f"{stats.images} images · skipped {stats.skipped}"
    )
    for error in stats.errors[:10]:
        console.print(f"[red]! {error}[/red]")
    if len(stats.errors) > 10:
        console.print(f"[red]... and {len(stats.errors) - 10} more errors[/red]")
    console.print(f"index: {db}")


@app.command("search")
def search_command(
    query: str = typer.Argument(...),
    db: Path = typer.Option(db_path(), help="SQLite index path."),
    limit: int = typer.Option(10, "--limit", "-n"),
    kind: str | None = typer.Option(None, help="Filter: text, pdf, or image."),
    since: str | None = typer.Option(None, "--since", help="Modified on or after this date (YYYY-MM-DD)."),
    until: str | None = typer.Option(None, "--until", help="Modified on or before this date (YYYY-MM-DD)."),
    min_size: str | None = typer.Option(None, "--min-size", help="Minimum file size, e.g. 10MB."),
    max_size: str | None = typer.Option(None, "--max-size", help="Maximum file size, e.g. 10MB."),
    folder: str | None = typer.Option(None, "--folder", help="Only files inside this folder prefix."),
    embedder: str = typer.Option("auto"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Search the index with natural language."""
    store = _store(db)
    try:
        results = run_search(
            store,
            build_embedder(embedder),
            query,
            limit=limit,
            kind=kind,
            since=since,
            until=until,
            min_size=min_size,
            max_size=max_size,
            folder=folder,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except IndexMismatch as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"Delete the index and run again:\n  rm {db}")
        raise typer.Exit(code=2) from exc
    if json_out:
        typer.echo(json.dumps([result.to_dict() for result in results], indent=2, ensure_ascii=False))
        return
    if not results:
        console.print("[yellow]No results. Is the folder indexed?[/yellow]")
        return
    table = Table(title=f'find-local: "{query}"', expand=False)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Kind", width=6)
    table.add_column("Path", max_width=52, overflow="ellipsis")
    table.add_column("Snippet", max_width=48, overflow="ellipsis")
    for result in results:
        table.add_row(
            f"{result.score:.3f}",
            result.kind,
            result.path,
            result.snippet[:90].replace("\n", " "),
        )
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8790),
    db: Path = typer.Option(db_path()),
) -> None:
    """Start the local search UI."""
    import uvicorn

    os.environ["FIND_DB"] = str(db)
    uvicorn.run("find_local.server:app", host=host, port=port, log_level="info")


@app.command()
def doctor() -> None:
    """Check models, torch, and index state."""
    import numpy
    import pymupdf

    console.print(f"pymupdf: [bold]{pymupdf.__version__}[/bold]")
    console.print(f"numpy: [bold]{numpy.__version__}[/bold]")
    try:
        import torch
        from sentence_transformers import SentenceTransformer  # noqa: F401

        console.print(f"torch: [bold]{torch.__version__}[/bold]  mps: {torch.backends.mps.is_available()}")
        console.print("[green]sentence-transformers ready[/green]")
    except ImportError:
        console.print("[yellow]sentence-transformers missing: uv pip install 'find-local[models]'[/yellow]")
    path = db_path()
    embedder = build_embedder("auto")
    console.print(f"embedder: [bold]{embedder.name}[/bold] ({embedder.dim}d)")
    if path.exists():
        store = Store(path)
        console.print(f"index: {path} · {store.stats()}")
        console.print(f"index model: {store.get_meta('embedder') or 'legacy or empty'}")
        store.close()
    else:
        console.print(f"index: {path} [yellow](not created yet)[/yellow]")


@app.command()
def watch(
    paths: list[Path] = typer.Argument(..., exists=True),
    db: Path = typer.Option(db_path(), help="SQLite index path."),
    embedder: str = typer.Option("auto"),
    interval: float = typer.Option(5.0, "--interval", min=0.5, help="Polling interval in seconds."),
) -> None:
    """Index now, then keep watching for changed files."""


    run_watch(paths, db, embedder, interval)


@app.command()
def mcp(db: Path = typer.Option(db_path())) -> None:
    """Run the MCP server over stdio."""
    os.environ["FIND_DB"] = str(db)
    try:
        from .mcp_server import main
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    main()
