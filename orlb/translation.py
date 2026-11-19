"""Shared logic for translating OR-Instruct coptpy solutions into PuLP / Pyomo and verifying them
against the coptpy reference objective. Used by scripts/s01_translate.py (local LLM) and by
api_translate/translate.py (Claude / Gemini APIs), so both produce identical data files."""
from __future__ import annotations

import os
import re
from collections import Counter

from .data import build_completion, math_part, save_jsonl, split_of
from .execution import NO_SOLUTION, feedback_message
from .grading import NO_SOL, match, pred_value
from .prompts import get_library, task_prompt, translate_prompt

_BLOCK = re.compile(r"```[ \t]*(?:python|py)[ \t]*\n(.*?)```", re.S | re.I)
_REFUSALS = ("NOT_LINEAR", "NOT_SUPPORTED")


def last_code_block(text):
    blocks = _BLOCK.findall(text or "")
    return blocks[-1] if blocks else None


def refused(text) -> bool:
    """Translator says the model is outside what the library supports (e.g. quadratic for PuLP)."""
    tail = (text or "").strip()[-200:]
    return any(r in tail for r in _REFUSALS) and last_code_block(text) is None


def looks_legit(code, library) -> bool:
    lib = get_library(library)
    return bool(code) and "OBJECTIVE_VALUE" in code and all(m in code for m in lib.required_markers)


def ref_answer(it):
    if it.get("ref_status") == "ok":
        return it["ref_objective"]
    if it.get("ref_status") in NO_SOLUTION:
        return NO_SOL
    return None


def init_items(items: list[dict], library: str, val_frac: float = 0.05) -> list[dict]:
    for it in items:
        it["answer"] = ref_answer(it)
        it["split"] = split_of(it["id"], val_frac)
        it.update(library=library, target_code=None, target_objective=None, verified_by=None,
                  refused=False, attempt_log=[])
    return items


def needs_translation(it, allow_agreement: bool = False) -> bool:
    return (it["target_code"] is None and not it["refused"] and bool(it.get("orig_code"))
            and (it["answer"] is not None or allow_agreement))


def record_attempt(it: dict, text: str, code, res, library: str, rel_tol: float = 1e-4,
                   allow_agreement: bool = False) -> bool:
    """Update item with one translation attempt. Returns True if the attempt was accepted."""
    if refused(text):
        it["refused"] = True
        it["attempt_log"].append({"status": "refused"})
        return False
    it["_last_code"] = code or ""
    legit = looks_legit(code, library)
    pv = pred_value(res.status, res.objective) if legit else None
    it["attempt_log"].append({"status": res.status if legit else ("no_code" if not code else "not_legit"),
                              "objective": res.objective if legit else None, "legit": legit,
                              "feedback": translation_feedback(code, res, library, it.get("answer"))})
    if pv is None:
        return False
    if it["answer"] is not None:
        if match(pv, it["answer"], rel_tol=rel_tol):
            it["target_code"], it["target_objective"], it["verified_by"] = code, res.objective, "reference"
            return True
    elif allow_agreement:
        prev = [a for a in it["attempt_log"][:-1] if a.get("legit") and pred_value(a["status"], a["objective"]) is not None]
        if any(match(pred_value(a["status"], a["objective"]), pv, rel_tol=rel_tol) for a in prev):
            it["target_code"], it["target_objective"], it["verified_by"] = code, res.objective, "agreement"
            it["answer"] = pv
            return True
    return False


def translation_feedback(code, res, library: str, answer) -> str:
    """What went wrong, in words the translator can act on. Never reveals the reference value
    (so the translator cannot simply print it)."""
    lib = get_library(library)
    if not code:
        return "No ```python code block was found in your answer."
    if not looks_legit(code, library):
        need = ", ".join(f"`{m}`" for m in lib.required_markers)
        return (f"The program must use {need} and report the result exactly with the given pattern "
                "(a line `OBJECTIVE_VALUE: <value>` taken from the solver).")
    if res.status == "ok":
        if answer == NO_SOL:
            return (f"The program reports OBJECTIVE_VALUE {res.objective}, but the original coptpy program finds "
                    "the model infeasible or unbounded. Some constraint or bound was changed.")
        return (f"The program ran and reported OBJECTIVE_VALUE {res.objective}, but this does NOT match the optimal "
                "value of the original coptpy program. Compare every variable bound and integrality, the objective "
                "sense and expression, and every constraint coefficient against the coptpy code.")
    if res.status in NO_SOLUTION:
        return (f"The solver reports the translated model as {res.status}, but the original coptpy program has an "
                "optimal solution. Some constraint, bound or sign was translated incorrectly.")
    return feedback_message(res)


FEEDBACK_SUFFIX = """

A previous translation attempt is shown below. It is NOT correct.
```python
{code}
```
Problem with that attempt:
{feedback}

Write a corrected, complete translation following all the requirements above. Return only one ```python code block."""


def feedback_prompt(it: dict, library: str) -> str:
    """Prompt for a retry: the original task plus the last failed attempt and what went wrong."""
    base = translate_prompt(it["question"], it["orig_code"], library)
    last = it["attempt_log"][-1] if it["attempt_log"] else None
    code = (it.get("_last_code") or "").strip()
    if not last or not code:
        return base
    return base + FEEDBACK_SUFFIX.format(code=code, feedback=last["feedback"])


def write_outputs(items: list[dict], library: str, out_dir: str) -> dict:
    sft = {"train": [], "val": []}
    for it in items:
        it.pop("_last_code", None)
        it["prompt"] = task_prompt(it["question"], library)
        it["completion"] = (build_completion(math_part(it["orig_completion"], library), it["target_code"], library)
                            if it["target_code"] else None)
        if it["completion"]:
            sft[it["split"]].append({"id": it["id"], "prompt": it["prompt"], "completion": it["completion"]})
    save_jsonl(items, os.path.join(out_dir, "or_instruct.jsonl"))
    for k, v in sft.items():
        save_jsonl(v, os.path.join(out_dir, f"sft_{k}.jsonl"))
    stats = {
        "verified_by": dict(Counter(it["verified_by"] for it in items)),
        "refused": sum(it["refused"] for it in items),
        "unverified_ref_status": dict(Counter(it.get("ref_status") for it in items if not it["target_code"])),
        "sft_train": len(sft["train"]), "sft_val": len(sft["val"]),
    }
    return stats
