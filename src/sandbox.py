"""Stage 2: safely execute untrusted generated solve() code.

Uses multiprocessing for cross-platform timeout enforcement (Windows laptop
+ Linux/Kaggle). Blocks dangerous imports and file/network access.
"""

from __future__ import annotations

import ast
import multiprocessing as mp
import os
import queue
import re
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


# Graduated timeout tiers (in seconds) for retry attempts (e.g. attempt 0 -> 5s, attempt 1 -> 8s, attempt 2+ -> 10s)
SANDBOX_TIMEOUT_TIERS: List[float] = [5.0, 8.0, 10.0]


def get_timeout_for_attempt(attempt_idx: int, tiers: Optional[Sequence[float]] = None) -> float:
    """Get the sandbox execution timeout (seconds) for an attempt index (0-based)."""
    active_tiers = tiers if tiers is not None else SANDBOX_TIMEOUT_TIERS
    if not active_tiers:
        return 5.0
    idx = min(max(0, attempt_idx), len(active_tiers) - 1)
    return float(active_tiers[idx])


@dataclass
class ExecutionResult:
    success: bool
    output_grid: Optional[Grid] = None
    error_message: Optional[str] = None
    execution_trace: List[str] = field(default_factory=list)
    partial_credit: Optional[float] = None  # fraction of cells matching expected
    is_timeout: bool = False


def cell_match_fraction(expected: Grid, actual: Optional[Grid]) -> float:
    """Fraction of cells that match. Shape mismatch → 0.0."""
    if actual is None or not expected or not actual:
        return 0.0
    eh, ew = len(expected), len(expected[0])
    ah, aw = len(actual), len(actual[0]) if actual else 0
    if (eh, ew) != (ah, aw) or eh * ew == 0:
        return 0.0
    match = sum(
        1 for r in range(eh) for c in range(ew) if expected[r][c] == actual[r][c]
    )
    return match / float(eh * ew)


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


def _persistent_worker_loop(task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Long-running worker process: executes code tasks in clean isolated namespaces."""
    import io
    import sys
    import traceback
    from typing import Any, List, Optional

    while True:
        try:
            task = task_queue.get()
        except (EOFError, KeyboardInterrupt):
            break
        if task is None:
            break

        task_id, code_str, input_grid = task
        trace: List[str] = []
        buf = io.StringIO()

        try:
            # Fresh namespace for EVERY task - ZERO state leakage across runs
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
                compiled = compile(code_str, "<generated>", "exec")
                exec(compiled, namespace, namespace)
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
            result_queue.put(
                {
                    "task_id": task_id,
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
            result_queue.put(
                {
                    "task_id": task_id,
                    "success": False,
                    "output_grid": None,
                    "error_message": f"{type(e).__name__}: {e}",
                    "execution_trace": trace,
                }
            )


def _worker(code_str: str, input_grid: Grid, queue: mp.Queue) -> None:
    """One-off child process target for standalone execution."""
    t_q: mp.Queue = mp.Queue()
    t_q.put((1, code_str, input_grid))
    t_q.put(None)
    _persistent_worker_loop(t_q, queue)


class _WorkerSlot:
    def __init__(self, ctx: mp.context.BaseContext):
        self.ctx = ctx
        self.task_queue: mp.Queue = ctx.Queue()
        self.result_queue: mp.Queue = ctx.Queue()
        self.process: mp.Process = ctx.Process(
            target=_persistent_worker_loop,
            args=(self.task_queue, self.result_queue),
            daemon=True,
        )
        self.process.start()

    def terminate_and_replace(self) -> None:
        try:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=0.2)
        except Exception:
            pass
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.process = self.ctx.Process(
            target=_persistent_worker_loop,
            args=(self.task_queue, self.result_queue),
            daemon=True,
        )
        self.process.start()


_TIMING_LOCK = threading.Lock()
_TIMING_COUNT = 0
_MAX_TIMING_COUNT = 20


def _record_timing(dispatch_ms: float, exec_ms: float) -> None:
    global _TIMING_COUNT
    with _TIMING_LOCK:
        if _TIMING_COUNT < _MAX_TIMING_COUNT:
            _TIMING_COUNT += 1
            tot = dispatch_ms + exec_ms
            print(f"[sandbox_timing] execution #{_TIMING_COUNT:02d}: dispatch_overhead={dispatch_ms:.3f}ms execution_time={exec_ms:.3f}ms total={tot:.3f}ms")


class SandboxPool:
    """Persistent worker pool for sandboxed code execution without per-task spawn overhead."""

    def __init__(self, num_workers: Optional[int] = None):
        if num_workers is None:
            cpu_cnt = os.cpu_count() or 4
            num_workers = max(1, min(cpu_cnt, 4))
        self.num_workers = num_workers
        self.ctx = mp.get_context("spawn")
        self.slots: List[_WorkerSlot] = [_WorkerSlot(self.ctx) for _ in range(num_workers)]
        self._available_slots: queue.Queue[int] = queue.Queue()
        for i in range(num_workers):
            self._available_slots.put(i)
        self._lock = threading.Lock()
        self._task_counter = 0

    def execute(
        self,
        code_str: str,
        input_grid: Grid,
        timeout_seconds: float = 5.0,
    ) -> ExecutionResult:
        t_dispatch_start = time.time()
        slot_idx = self._available_slots.get()
        slot = self.slots[slot_idx]
        dispatch_time_ms = (time.time() - t_dispatch_start) * 1000.0

        with self._lock:
            self._task_counter += 1
            task_id = self._task_counter

        t_exec_start = time.time()
        try:
            if not slot.process.is_alive():
                slot.terminate_and_replace()

            slot.task_queue.put((task_id, code_str, input_grid))

            try:
                raw_res = slot.result_queue.get(timeout=timeout_seconds)
                exec_time_ms = (time.time() - t_exec_start) * 1000.0
                _record_timing(dispatch_time_ms, exec_time_ms)
                return ExecutionResult(
                    success=bool(raw_res["success"]),
                    output_grid=raw_res.get("output_grid"),
                    error_message=raw_res.get("error_message"),
                    execution_trace=list(raw_res.get("execution_trace") or []),
                )
            except queue.Empty:
                exec_time_ms = (time.time() - t_exec_start) * 1000.0
                _record_timing(dispatch_time_ms, exec_time_ms)
                is_alive = slot.process.is_alive()
                slot.terminate_and_replace()
                err = f"Timeout after {timeout_seconds}s" if is_alive else "Worker process terminated unexpectedly"
                return ExecutionResult(
                    success=False,
                    output_grid=None,
                    error_message=err,
                    execution_trace=[err],
                    is_timeout=is_alive,
                )
        finally:
            self._available_slots.put(slot_idx)

    def shutdown(self) -> None:
        for slot in self.slots:
            try:
                slot.task_queue.put(None)
            except Exception:
                pass
            slot.terminate_and_replace()


_GLOBAL_POOL: Optional[SandboxPool] = None
_POOL_LOCK = threading.Lock()


def get_sandbox_pool(num_workers: Optional[int] = None) -> SandboxPool:
    """Retrieve or lazily create the global persistent sandbox pool."""
    global _GLOBAL_POOL
    with _POOL_LOCK:
        if _GLOBAL_POOL is None:
            _GLOBAL_POOL = SandboxPool(num_workers=num_workers)
        return _GLOBAL_POOL


def init_sandbox_pool(num_workers: Optional[int] = None) -> SandboxPool:
    """Explicitly initialize the persistent sandbox pool (e.g. at startup)."""
    return get_sandbox_pool(num_workers=num_workers)


def shutdown_sandbox_pool() -> None:
    """Cleanly shut down all background worker processes in the sandbox pool."""
    global _GLOBAL_POOL
    with _POOL_LOCK:
        if _GLOBAL_POOL is not None:
            _GLOBAL_POOL.shutdown()
            _GLOBAL_POOL = None


def validate_function_calls(
    code_str: str,
    allowed_names: Optional[set[str]] = None,
) -> List[str]:
    """Return a list of called function names not in builtins, primitives, or local scope."""
    code_str = _strip_markdown_fence(code_str)
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return []

    if allowed_names is None:
        try:
            from src.library import list_primitive_names

            allowed_names = set(list_primitive_names())
        except Exception:
            allowed_names = set()

    builtins_set = set(_safe_builtins().keys()) | {
        "List",
        "Tuple",
        "Optional",
        "Any",
        "ObjectInfo",
        "AxisInfo",
        "Direction",
        "Grid",
    }

    local_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local_names.add(node.name)
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                local_names.add(arg.arg)
            if node.args.vararg:
                local_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                local_names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                local_names.add(arg.arg)
            if node.args.vararg:
                local_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                local_names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            local_names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local_names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            local_names.add(node.name)

    unknown: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if (
                    fname not in builtins_set
                    and fname not in allowed_names
                    and fname not in local_names
                ):
                    if fname not in unknown:
                        unknown.append(fname)
            elif isinstance(node.func, ast.Attribute):
                curr = node.func
                parts = [node.func.attr]
                while isinstance(curr.value, ast.Attribute):
                    curr = curr.value
                    parts.append(curr.attr)
                if isinstance(curr.value, ast.Name):
                    root_id = curr.value.id
                    if (
                        root_id not in builtins_set
                        and root_id not in allowed_names
                        and root_id not in local_names
                    ):
                        full_name = (
                            f"{root_id}.{parts[0]}"
                            if len(parts) == 1
                            else f"{root_id}.{'.'.join(reversed(parts[:-1]))}"
                        )
                        if full_name not in unknown:
                            unknown.append(full_name)
    return unknown


def safe_execute(
    code_str: str,
    input_grid: Grid,
    timeout_seconds: float = 5.0,
    pool: Optional[SandboxPool] = None,
) -> ExecutionResult:
    """Run untrusted code with a hard timeout via persistent worker pool. Never raises to the caller."""
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

    unknown = validate_function_calls(code_str)
    if unknown:
        err = f"Unknown function(s) {unknown} - not in library.py, not a builtin, not defined locally."
        return ExecutionResult(
            success=False,
            error_message=err,
            execution_trace=[err],
        )

    active_pool = pool if pool is not None else get_sandbox_pool()
    return active_pool.execute(code_str, input_grid, timeout_seconds=timeout_seconds)
