"""Command line interface."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import db_path
from .embed import build_embedder
from .index import Indexer, Store
from .search import search as run_search

app = typer.Typer(add_completion=False, help="Search your computer by describing it — offline.")
console = Console(width=130)


def _store(path: Path) -> Store:
    return Store(path)


@app.command("index")
def index_command(
    paths: list[Path] = typer.Argument(..., exists=True, help="Files or folders to index."),
    db: Path = typer.Option(db_path(), help="SQLite index path."),
    embedder: str = typer.Option("auto", help="auto or hash (tests)."),
    no_images: bool = typer.Option(False, "--no-images", help="Skip image embeddings."),
) -> None:
    """Index files, PDFs, and images."""
    store = _store(db)
    indexer = Indexer(store, build_embedder(embedder, with_images=not no_images))
    stats = indexer.index_paths(paths)
    console.print(
        f"[green]indexed[/green] {stats.files} files · {stats.chunks} chunks · "
        f"{stats.images} images · skipped {stats.skipped}"
    )
    for error in stats.errors[:10]:
        console.print(f"[red]! {error}[/red]")
    if len(stats.errors) > 10:
        console.print(f"[red]… and {len(stats.errors) - 10} more errors[/red]")
    console.print(f"index: {db}")


@app.command("search")
def search_command(
    query: str = typer.Argument(...),
    db: Path = typer.Option(db_path(), help="SQLite index path."),
    limit: int = typer.Option(10, "--limit", "-n"),
    kind: str | None = typer.Option(None, help="Filter: text, pdf, or image."),
    embedder: str = typer.Option("auto"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Search the index with natural language."""
    store = _store(db)
    results = run_search(store, build_embedder(embedder), query, limit=limit, kind=kind)
    if json_out:
        typer.echo(json.dumps([result.to_dict() for result in results], indent=2, ensure_ascii=False))
        return
    if not results:
        console.print("[yellow]No results. Is the folder indexed?[/yellow]")
        return
    table = Table(title=f"find-local · “{query}”", expand=False)
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
    if path.exists():
        store = Store(path)
        console.print(f"index: {path} · {store.stats()}")
        store.close()
    else:
        console.print(f"index: {path} [yellow](not created yet)[/yellow]")


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
