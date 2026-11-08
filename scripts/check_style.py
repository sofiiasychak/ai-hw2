#!/usr/bin/env python
"""Check one-line one-sentence docstrings and short comments in code and notebooks."""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path

from hw2.paths import repo_root

MAX_COMMENT_WORDS = 5
CODE_DIRS = ("src", "scripts", "tests")
NOTEBOOK_GLOB = "notebooks/*.ipynb"
DOC_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
SENTENCE_BREAK = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)[.!?]\s+[A-Z]")
NOQA = re.compile(r"\bnoqa\b")
MAGIC_LINE = re.compile(r"^\s*[%!]")
PYTHON_CELL_MAGICS = ("%%time", "%%timeit", "%%capture")


def docstring_issues(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Yield line and message for each docstring spanning several lines or sentences."""
    for node in ast.walk(tree):
        if not isinstance(node, DOC_NODES):
            continue
        text = ast.get_docstring(node, clean=False)
        if text is None:
            continue
        expr = node.body[0]
        if expr.end_lineno != expr.lineno or "\n" in text:
            yield expr.lineno, "docstring spans more than one line"
        if SENTENCE_BREAK.search(text.strip()):
            yield expr.lineno, "docstring has more than one sentence"


def _skip_comment(tok: tokenize.TokenInfo) -> bool:
    """Return True for shebang, noqa and type comments."""
    body = tok.string.lstrip("#").strip()
    is_shebang = tok.start[0] == 1 and tok.string.startswith("#!")
    return is_shebang or body.startswith("type:") or bool(NOQA.search(body))


def comment_issues(source: str) -> Iterator[tuple[int, str]]:
    """Yield line and message for each comment with more words than allowed."""
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.COMMENT or _skip_comment(tok):
            continue
        words = len(tok.string.lstrip("#").split())
        if words > MAX_COMMENT_WORDS:
            yield tok.start[0], f"comment has {words} words (max {MAX_COMMENT_WORDS})"


def check_source(source: str) -> list[tuple[int, str]]:
    """Return line and message for every style issue in one Python source string."""
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        return [(err.lineno or 0, f"syntax error: {err.msg}")]
    return sorted([*docstring_issues(tree), *comment_issues(source)])


def notebook_cells(path: Path) -> Iterator[tuple[int, str]]:
    """Yield index and magic-free source of each Python code cell in a notebook."""
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    for index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        head = source.lstrip()
        if head.startswith("%%") and not head.startswith(PYTHON_CELL_MAGICS):
            continue
        lines = ("" if MAGIC_LINE.match(line) else line for line in source.splitlines())
        yield index, "\n".join(lines) + "\n"


def iter_sources(root: Path) -> Iterator[tuple[str, str]]:
    """Yield a display label and Python source for every file and notebook cell to check."""
    for folder in CODE_DIRS:
        for path in sorted((root / folder).rglob("*.py")):
            yield str(path.relative_to(root)), path.read_text(encoding="utf-8")
    for path in sorted(root.glob(NOTEBOOK_GLOB)):
        for index, source in notebook_cells(path):
            yield f"{path.relative_to(root)}:cell{index}", source


def main() -> int:
    """Print every style issue as file:line and return 1 when any is found."""
    root = repo_root()
    issues = [
        f"{label}:{line}: {message}"
        for label, source in iter_sources(root)
        for line, message in check_source(source)
    ]
    for issue in issues:
        print(issue)
    print(f"check_style: {len(issues)} issue(s)" if issues else "check_style: ok")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
