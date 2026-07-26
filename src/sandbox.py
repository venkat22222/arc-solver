"""Stage 2: safely execute untrusted generated solve() code.

Uses multiprocessing for cross-platform timeout enforcement (Windows laptop
+ Linux/Kaggle). Blocks dangerous imports and file/network access.
"""

from __future__ import annotations

import ast
import multiprocessing as mp
import re
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional

Grid = List[List[int]]

_BLOCKED_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "pickle",
        "ctypes",
        "importlib",
        "builtins",
        "multiprocessing",
        "threading",
        "signal",
        "pty",
        "fcntl",
        "resource",
    }
)

_BLOCKED_BUILTINS = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "memoryview",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    }
)


@dataclass
class ExecutionResult:
    success: bool
    output_grid: Optional[Grid] = None
    error_message: Optional[str] = None
    execution_trace: List[str] = field(default_factory=list)


class _SecurityError(Exception):
    pass


class _ImportGuard(ast.NodeVisitor):
    """Reject AST nodes that import or call blocked modules/builtins."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _BLOCKED_IMPORTS:
                raise _SecurityError(f"Blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in _BLOCKED_IMPORTS:
                raise _SecurityError(f"Blocked import from: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_BUILTINS:
            raise _SecurityError(f"Blocked builtin call: {node.func.id}")
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            raise _SecurityError("Blocked __import__")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block os.system-style attribute access on known dangerous names
        if isinstance(node.value, ast.Name) and node.value.id in _BLOCKED_IMPORTS:
            raise _SecurityError(f"Blocked attribute access on {node.value.id}")
        self.generic_visit(node)


def validate_code_safety(code_str: str) -> None:
    """Raise SecurityError if code uses blocked imports/builtins."""
    # Strip markdown fences if the model wrapped the code
    code_str = _strip_markdown_fence(code_str)
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in generated code: {e}") from e
    _ImportGuard().visit(tree)


def _strip_markdown_fence(code_str: str) -> str:
    code_str = code_str.strip()
    fence = re.match(r"^```(?:python)?\s*\n(.*?)```\s*$", code_str, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return code_str


def _safe_builtins() -> dict:
    safe = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "frozenset": frozenset,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "print": print,  # captured via redirected stdout in worker
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    }
    return safe


def _inject_library_helpers(namespace: dict[str, Any]) -> None:
    """Load DreamCoder primitives into the sandbox globals (best-effort)."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.library import get_sandbox_helpers

        namespace.update(get_sandbox_helpers())
    except Exception:
        # Library unavailable in this environment — generated code must be self-contained.
        pass


def _worker(code_str: str, input_grid: Grid, queue: mp.Queue) -> None:
    """Child process target — execute solve(grid) and push result."""
    import io
    import sys
    from typing import List as TypingList  # noqa: F401 — available to generated code

    trace: List[str] = []
    buf = io.StringIO()

    try:
        validate_code_safety(code_str)
        code_str = _strip_markdown_fence(code_str)

        # Allow typing.List in annotations without a full typing import surface
        namespace: dict[str, Any] = {
            "__builtins__": _safe_builtins(),
            "List": list,
            "Tuple": tuple,
            "Optional": Optional,
            "Any": Any,
        }
        _inject_library_helpers(namespace)

        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            exec(compile(code_str, "<generated>", "exec"), namespace, namespace)
            if "solve" not in namespace or not callable(namespace["solve"]):
                raise RuntimeError("Generated code must define solve(grid)")
            # Deep-copy input so mutations don't confuse callers
            grid_copy = [row[:] for row in input_grid]
            result = namespace["solve"](grid_copy)
        finally:
            sys.stdout = old_stdout

        printed = buf.getvalue()
        if printed:
            trace.extend(printed.rstrip("\n").split("\n"))

        if not isinstance(result, list) or not result or not isinstance(result[0], list):
            raise TypeError(f"solve() must return List[List[int]], got {type(result)}")
        out: Grid = [[int(c) for c in row] for row in result]
        queue.put(
            {
                "success": True,
                "output_grid": out,
                "error_message": None,
                "execution_trace": trace,
            }
        )
    except Exception as e:
        printed = buf.getvalue()
        if printed:
            trace.extend(printed.rstrip("\n").split("\n"))
        trace.append(traceback.format_exc())
        queue.put(
            {
                "success": False,
                "output_grid": None,
                "error_message": f"{type(e).__name__}: {e}",
                "execution_trace": trace,
            }
        )


def safe_execute(
    code_str: str,
    input_grid: Grid,
    timeout_seconds: float = 5.0,
) -> ExecutionResult:
    """Run untrusted code with a hard timeout. Never raises to the caller."""
    code_str = textwrap.dedent(_strip_markdown_fence(code_str))

    # Fast-fail safety check in parent (also re-checked in child)
    try:
        validate_code_safety(code_str)
    except (_SecurityError, SyntaxError) as e:
        return ExecutionResult(
            success=False,
            error_message=str(e),
            execution_trace=[str(e)],
        )

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code_str, input_grid, queue))
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)
        return ExecutionResult(
            success=False,
            error_message=f"Timeout after {timeout_seconds}s",
            execution_trace=[f"Timeout after {timeout_seconds}s"],
        )

    if queue.empty():
        return ExecutionResult(
            success=False,
            error_message="Worker exited without returning a result",
            execution_trace=[],
        )

    payload = queue.get()
    return ExecutionResult(
        success=bool(payload["success"]),
        output_grid=payload.get("output_grid"),
        error_message=payload.get("error_message"),
        execution_trace=list(payload.get("execution_trace") or []),
    )
