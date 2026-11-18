"""Generate -> execute -> grade -> (optionally) repair. Shared by sampling (training data) and evaluation."""
from __future__ import annotations

from typing import Optional

from .execution import REPAIRABLE, NO_SOLUTION, extract_code, feedback_message, run_many
from .grading import match, majority_vote, parse_gt, pred_value
from .modeling import GenStats, Generator
from .prompts import repair_prompt


def solve_items(gen: Generator, items: list[dict], prompts: list[str], n: int, temperature: float,
                top_p: float = 0.95, seed: int = 0, repair_rounds: int = 0, repair_no_solution: bool = False,
                exec_timeout: float = 60.0, exec_workers: Optional[int] = None, rel_tol: float = 1e-4) -> tuple[list[dict], dict]:
    """items: [{id, answer, ...}]. Returns per-item records and cost stats.

    Each record has `samples`: list of {attempts: [{response, status, objective, ...}], final: {...}}.
    """
    gen_stats, rep_stats = GenStats(), GenStats()
    responses, st = gen.generate(prompts, n=n, temperature=temperature, top_p=top_p, seed=seed)
    gen_stats.add(st)

    flat = [(i, s) for i in range(len(items)) for s in range(len(responses[i]))]
    results = run_many([extract_code(responses[i][s]) for i, s in flat], timeout=exec_timeout, workers=exec_workers)
    samples = [[None] * len(responses[i]) for i in range(len(items))]
    for (i, s), r in zip(flat, results):
        samples[i][s] = {"attempts": [_attempt(responses[i][s], r, prompts[i], kind="initial")]}

    repair_status = set(REPAIRABLE) | (set(NO_SOLUTION) if repair_no_solution else set())
    for rnd in range(repair_rounds):
        todo = [(i, s) for i in range(len(items)) for s in range(len(samples[i]))
                if samples[i][s]["attempts"][-1]["status"] in repair_status]
        if not todo:
            break
        rprompts = []
        for i, s in todo:
            last = samples[i][s]["attempts"][-1]
            rprompts.append(repair_prompt(prompts[i], last["response"], last["feedback"]))
        rresp, st = gen.generate(rprompts, n=1, temperature=temperature if temperature > 0 else 0.0,
                                 top_p=top_p, seed=seed + 7919 * (rnd + 1))
        rep_stats.add(st)
        rres = run_many([extract_code(r[0]) for r in rresp], timeout=exec_timeout, workers=exec_workers)
        for (i, s), rp, resp, res in zip(todo, rprompts, rresp, rres):
            samples[i][s]["attempts"].append(_attempt(resp[0], res, rp, kind=f"repair{rnd + 1}"))

    records = []
    for i, it in enumerate(items):
        gt, dec = parse_gt(it.get("answer"))
        for smp in samples[i]:
            for a in smp["attempts"]:
                pv = pred_value(a["status"], a["objective"])
                a["correct"] = match(pv, gt, dec, rel_tol)
                a["correct_lenient"] = match(pv, gt, dec, rel_tol, lenient=True)
            smp["first"] = smp["attempts"][0]
            smp["final"] = smp["attempts"][-1]
        rec = {k: v for k, v in it.items()}
        rec["gt_parsed"] = gt
        rec["samples"] = samples[i]
        records.append(rec)
    stats = {"generation": gen_stats.to_dict(), "repair": rep_stats.to_dict()}
    return records, stats


def _attempt(response: str, res, prompt: str, kind: str) -> dict:
    return {
        "kind": kind,
        "prompt": prompt if kind != "initial" else None,
        "response": response,
        "status": res.status,
        "objective": res.objective,
        "parse_mode": res.parse_mode,
        "error_type": res.error_type,
        "feedback": feedback_message(res),
        "stdout_tail": res.stdout[-800:],
        "stderr_tail": res.stderr[-800:],
        "exec_seconds": round(res.seconds, 3),
    }


def score_records(records: list[dict], ks=(1, 2, 4, 8, 16), rel_tol: float = 1e-4, which: str = "final") -> dict:
    """Accuracy metrics over a set of records.

    mean@1  : average single-sample accuracy (all samples)
    maj@k   : majority vote over the first k samples
    pass@k  : any of the first k samples correct (oracle upper bound; diagnostics only)
    """
    n = len(records)
    out = {"n_problems": n}
    if n == 0:
        return out
    n_samp = min(len(r["samples"]) for r in records)
    for key in ("correct", "correct_lenient"):
        tag = "" if key == "correct" else "_lenient"
        out[f"mean@1{tag}"] = sum(sum(s[which][key] for s in r["samples"]) / len(r["samples"]) for r in records) / n
        out[f"first_sample_acc{tag}"] = sum(r["samples"][0][which][key] for r in records) / n
        for k in ks:
            if k > n_samp:
                continue
            out[f"pass@{k}{tag}"] = sum(any(s[which][key] for s in r["samples"][:k]) for r in records) / n
            if k > 1:
                hits = 0
                for r in records:
                    preds = [pred_value(s[which]["status"], s[which]["objective"]) for s in r["samples"][:k]]
                    win = majority_vote(preds, rel_tol)
                    gt, dec = parse_gt(r.get("answer"))
                    hits += match(win, gt, dec, rel_tol, lenient=(key != "correct"))
                out[f"maj@{k}{tag}"] = hits / n
    # error breakdown of first samples
    br = {}
    for r in records:
        a = r["samples"][0][which]
        cat = "correct" if a["correct"] else ("wrong_value" if a["status"] == "ok" else a["status"])
        br[cat] = br.get(cat, 0) + 1
    out["breakdown_first_sample"] = br
    modes = {}
    for r in records:
        for s in r["samples"]:
            m = s[which]["parse_mode"] or "none"
            modes[m] = modes.get(m, 0) + 1
    out["parse_modes"] = modes
    return out
