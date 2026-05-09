"""
Model Context Protocol (stdio) server for read-only access to this repository.
Configure in Cursor MCP settings to point at this file with: python mcp_codebase.py
"""
from pathlib import Path

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent
IGNORE_NAMES = {".git", "__pycache__", "node_modules", ".venv", "venv", ".cursor"}

mcp = FastMCP("Ensemble Codebase", json_response=True)


def _is_under_project(path: Path) -> bool:
    try:
        path.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def _resolve_project_path(relative_path: str) -> Path:
    rel = (relative_path or ".").strip() or "."
    candidate = (PROJECT_ROOT / rel).resolve()
    if not _is_under_project(candidate):
        raise ValueError("Path escapes project root")
    return candidate


@mcp.tool()
def read_project_file(relative_path: str) -> str:
    """Read a UTF-8 text file under the project root. Use paths like main.py or src/app.py."""
    try:
        path = _resolve_project_path(relative_path)
    except ValueError as e:
        return f"Error: {e}"
    if not path.is_file():
        return f"Error: not a file: {relative_path}"
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading file: {e}"


@mcp.tool()
def list_project_files(relative_directory: str = ".", max_entries: int = 400) -> str:
    """List files and directories relative to project root (skips common ignored folders). Caps output size."""
    try:
        base = _resolve_project_path(relative_directory)
    except ValueError as e:
        return f"Error: {e}"
    if not base.is_dir():
        return f"Error: not a directory: {relative_directory}"
    lines: list[str] = []
    cap = max(50, min(max_entries, 2000))
    n = 0
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(PROJECT_ROOT)
        if any(part in IGNORE_NAMES for part in rel.parts):
            continue
        n += 1
        if n > cap:
            lines.append(f"... truncated after {cap} entries")
            break
        kind = "dir" if p.is_dir() else "file"
        lines.append(f"{kind}\t{rel.as_posix()}")
    return "\n".join(lines) if lines else "(empty)"


@mcp.resource("codebase://project_root")
def project_root_uri() -> str:
    """Absolute path to the repository root."""
    return str(PROJECT_ROOT)


if __name__ == "__main__":
    mcp.run(transport="stdio")
