"""Extract generated code, run it in a subprocess with a timeout, parse the result."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

# Statuses that a repair round can try to fix.
REPAIRABLE = {"no_code", "syntax_error", "runtime_error", "timeout", "no_objective"}
NO_SOLUTION = {"infeasible", "unbounded", "infeasible_or_unbounded"}

_NUM = r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_TAGGED_BLOCK = re.compile(r"```[ \t]*(?:python|py|python3)[ \t]*\n(.*?)```", re.S | re.I)
_ANY_BLOCK = re.compile(r"```[ \t]*\n(.*?)```", re.S)
_UNCLOSED = re.compile(r"```[ \t]*(?:python|py|python3)[ \t]*\n(.*)$", re.S | re.I)
_OBJ_MARKER = re.compile(r"OBJECTIVE_VALUE\s*[:=]\s*(" + _NUM + ")")
_STATUS_MARKER = re.compile(r"STATUS\s*[:=]\s*([A-Za-z_\.]+)")
_ERR_LINE = re.compile(r"^([A-Za-z_][\w\.]*(?:Error|Exception|Exit|Interrupt))\b", re.M)
_KEYWORDS = re.compile(
    r"objective|optimal value|optimum|total|minimum|maximum|minimi|maximi|cost|profit|revenue|"
    r"objval|obj val",
    re.I,
)


@dataclass
class ExecResult:
    status: str  # ok | no_code | syntax_error | runtime_error | timeout | no_objective | infeasible | unbounded | infeasible_or_unbounded
    objective: Optional[float] = None
    parse_mode: Optional[str] = None  # marker | keyword | last_number
    error_type: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @property
    def has_answer(self) -> bool:
        return self.status == "ok" or self.status in NO_SOLUTION


def extract_code(text: str) -> Optional[str]:
    """Return the first python code block. Falls back to an untagged or unclosed block."""
    if not text:
        return None
    m = _TAGGED_BLOCK.search(text)
    if m:
        return m.group(1)
    m = _ANY_BLOCK.search(text)
    if m and ("import" in m.group(1) or "pyo." in m.group(1)):
        return m.group(1)
    m = _UNCLOSED.search(text)  # generation stopped inside the block
    if m:
        return m.group(1)
    return None


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def _normalise_status(raw: str) -> str:
    s = raw.lower().replace("terminationcondition.", "").strip(". ")
    if "infeasibleorunbounded" in s.replace("_", "") or ("infeasible" in s and "unbounded" in s):
        return "infeasible_or_unbounded"
    if "infeasible" in s:
        return "infeasible"
    if "unbounded" in s:
        return "unbounded"
    return "no_objective"


def parse_output(stdout: str, stderr: str, returncode: int) -> ExecResult:
    m = _OBJ_MARKER.findall(stdout)
    if m:
        val = _to_float(m[-1])
        if val is not None:
            return ExecResult("ok", val, "marker")
    m = _STATUS_MARKER.findall(stdout)
    if m:
        return ExecResult(_normalise_status(m[-1]), None, "marker")
    if returncode != 0:
        et = _ERR_LINE.findall(stderr)
        return ExecResult("runtime_error", error_type=et[-1] if et else f"exit_{returncode}")
    # Fallbacks for programs that ignore the convention (e.g. coptpy reference code).
    lines = [l for l in stdout.splitlines() if l.strip()]
    for line in reversed(lines):
        if _KEYWORDS.search(line):
            nums = re.findall(_NUM, line)
            if nums:
                val = _to_float(nums[-1])
                if val is not None:
                    return ExecResult("ok", val, "keyword")
    for line in lines:
        low = line.lower()
        if "infeasible" in low or "unbounded" in low:
            return ExecResult(_normalise_status(low), None, "keyword")
        if "no optimal" in low or "no feasible" in low or "no solution" in low:
            return ExecResult("infeasible_or_unbounded", None, "keyword")
    for line in reversed(lines):
        nums = re.findall(_NUM, line)
        if nums:
            val = _to_float(nums[-1])
            if val is not None:
                return ExecResult("ok", val, "last_number")
    return ExecResult("no_objective")


def run_code(code: Optional[str], timeout: float = 60.0, python: Optional[str] = None,
             prelude: str = "", postlude: str = "", max_output_chars: int = 20000) -> ExecResult:
    if code is None or not code.strip():
        return ExecResult("no_code")
    try:
        compile(code, "<generated>", "exec")
    except (SyntaxError, ValueError) as e:  # ValueError: null bytes
        return ExecResult("syntax_error", error_type=type(e).__name__, stderr=str(e))
    python = python or sys.executable
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="orlb_exec_") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(prelude + code + "\n" + postlude)
        proc = subprocess.Popen([python, path], cwd=d, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
            return ExecResult("timeout", stdout=out.decode("utf-8", "replace")[-max_output_chars:],
                              stderr=err.decode("utf-8", "replace")[-max_output_chars:], seconds=time.time() - t0)
    stdout = out.decode("utf-8", "replace")[-max_output_chars:]
    stderr = err.decode("utf-8", "replace")[-max_output_chars:]
    res = parse_output(stdout, stderr, proc.returncode)
    res.stdout, res.stderr, res.seconds = stdout, stderr, time.time() - t0
    return res


def run_many(codes, timeout: float = 60.0, workers: Optional[int] = None, **kw) -> list[ExecResult]:
    workers = workers or max(1, min(16, (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda c: run_code(c, timeout=timeout, **kw), codes))


def feedback_message(res: ExecResult, max_chars: int = 1500) -> str:
    if res.status == "no_code":
        return "No ```python code block was found in the response."
    if res.status == "syntax_error":
        return f"SyntaxError: {res.stderr[-max_chars:]}"
    if res.status == "timeout":
        return "The program did not finish within the time limit."
    if res.status == "runtime_error":
        return "The program crashed with this error:\n" + res.stderr[-max_chars:]
    if res.status == "no_objective":
        return ("The program ran but did not print a line of the form `OBJECTIVE_VALUE: <number>`.\n"
                "Program output:\n" + res.stdout[-max_chars:])
    if res.status in NO_SOLUTION:
        return f"The solver reported the model as {res.status}. Check the constraints and the variable bounds."
    return ""
