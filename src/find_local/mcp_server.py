"""MCP server exposing the local index to agents."""

from __future__ import annotations

import os
from pathlib import Path

from .config import db_path
from .embed import build_embedder
from .index import Indexer, Store
from .search import search as run_search


def _store() -> Store:
    return Store(Path(os.environ.get("FIND_DB", str(db_path()))))


def main() -> None:
    try:
        from mcp.server.mcpserver import MCPServer  # MCP SDK 2.x

        server = MCPServer("find-local")
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP  # MCP SDK 1.x

            server = FastMCP("find-local")
        except ImportError as exc:
            raise ImportError(
                "The MCP SDK is not installed. Install it with: uv pip install 'find-local[mcp]'"
            ) from exc

    @server.tool()
    def search_files(query: str, limit: int = 5) -> list[dict]:
        """Search indexed files, PDFs, and images with a natural-language query."""
        store = _store()
        try:
            return [result.to_dict() for result in run_search(store, build_embedder("auto"), query, limit=limit)]
        finally:
            store.close()

    @server.tool()
    def index_folder(path: str) -> dict:
        """Index a folder (recursively) into the local search index."""
        store = _store()
        try:
            stats = Indexer(store, build_embedder("auto")).index_paths([Path(path)])
            return {
                "files": stats.files,
                "chunks": stats.chunks,
                "images": stats.images,
                "skipped": stats.skipped,
                "errors": stats.errors[:5],
            }
        finally:
            store.close()

    @server.tool()
    def index_stats() -> dict:
        """Return counts for the current index."""
        store = _store()
        try:
            return store.stats()
        finally:
            store.close()

    server.run()
