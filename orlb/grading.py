"""Compare executed answers with ground truth, majority voting, simple statistics."""
from __future__ import annotations

import math
import re
from typing import Optional, Sequence, Union

from .execution import NO_SOLUTION

NO_SOL = "no_solution"
_NUM = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_NO_SOL_WORDS = ("infeasible", "no best solution", "no feasible", "unbounded", "no solution", "no optimal")


def parse_gt(x) -> tuple[Optional[Union[float, str]], int]:
    """Return (value, decimals). value is a float, NO_SOL, or None if unusable."""
    if x is None:
        return None, 0
    if isinstance(x, bool):
        return None, 0
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return None, 0
        return float(x), _decimals(repr(x))
    s = str(x).strip()
    if s == NO_SOL:
        return NO_SOL, 0
    if any(w in s.lower() for w in _NO_SOL_WORDS):
        return NO_SOL, 0
    m = _NUM.search(s)
    if not m:
        return None, 0
    tok = m.group(0).replace(",", "")
    try:
        return float(tok), _decimals(tok)
    except ValueError:
        return None, 0


def _decimals(s: str) -> int:
    s = s.lower()
    if "e" in s:
        return 6
    if "." not in s:
        return 0
    frac = s.split(".", 1)[1].rstrip("0")
    return min(len(frac), 6)


def pred_value(status: str, objective: Optional[float]):
    if status == "ok" and objective is not None and math.isfinite(objective):
        return objective
    if status in NO_SOLUTION:
        return NO_SOL
    return None


def match(pred, gt, gt_decimals: int = 0, rel_tol: float = 1e-4, lenient: bool = False) -> bool:
    """strict: |p-g| <= max(rel_tol*|g|, 1e-6).
    lenient: additionally accept p rounded to the precision the ground truth is written with."""
    if pred is None or gt is None:
        return False
    if gt == NO_SOL or pred == NO_SOL:
        return gt == pred
    if abs(pred - gt) <= max(rel_tol * abs(gt), 1e-6):
        return True
    if lenient:
        return round(pred, gt_decimals) == round(gt, gt_decimals)
    return False


def majority_vote(preds: Sequence, rel_tol: float = 1e-4):
    """preds: list of float | NO_SOL | None. Returns the winning answer or None."""
    clusters: list[list] = []  # [representative, count, first_index]
    for i, p in enumerate(preds):
        if p is None:
            continue
        for c in clusters:
            r = c[0]
            if (p == NO_SOL and r == NO_SOL) or (
                p != NO_SOL and r != NO_SOL and abs(p - r) <= max(rel_tol * abs(r), 1e-6)
            ):
                c[1] += 1
                break
        else:
            clusters.append([p, 1, i])
    if not clusters:
        return None
    clusters.sort(key=lambda c: (-c[1], c[2]))
    return clusters[0][0]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Paired test on the same problems. Returns (#a-only-correct, #b-only-correct, two-sided p)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)
