"""Step 7: error taxonomy.

Automatic coarse categories (no code / syntax / runtime / timeout / infeasible / wrong value) plus
cheap heuristic flags for wrong-value cases, and a CSV for manual labelling of a random subset.

python scripts/s07_error_analysis.py --runs results/sft_nf4_greedy results/sft_bf16_greedy --bench nl4opt
"""
import argparse
import csv
import math
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orlb.data import load_jsonl  # noqa: E402
from orlb.execution import extract_code  # noqa: E402
from orlb.grading import NO_SOL, parse_gt  # noqa: E402

MANUAL_COLS = ["wrong_variable_type", "missing_constraint", "extra_constraint", "wrong_coefficient",
               "wrong_objective_sense", "wrong_objective_expr", "misread_question", "ground_truth_suspect", "other"]


def flags(gt, pred, code):
    f = []
    if not isinstance(gt, float) or pred is None:
        return f
    code = code or ""
    if abs(gt - round(gt)) < 1e-6 and abs(pred - round(pred)) > 1e-4 and not re.search(r"Integers|Binary|Integer", code):
        f.append("integrality_missing?")
    if gt != 0 and abs(abs(pred) - abs(gt)) <= 1e-4 * abs(gt) and math.copysign(1, pred) != math.copysign(1, gt):
        f.append("sign_flip?")
    if gt != 0 and pred != 0:
        ratio = abs(pred / gt)
        for p in (10, 100, 1000):
            if abs(ratio - p) / p < 1e-3 or abs(ratio - 1 / p) * p < 1e-3:
                f.append("scale_x%d?" % p)
    if gt == 0 or pred == 0:
        f.append("zero_solution?")
    return f


def category(a):
    if a["correct"]:
        return "correct"
    if a["status"] == "ok":
        return "wrong_value"
    return a["status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--which", default="first", choices=["first", "final"])
    ap.add_argument("--manual_n", type=int, default=60, help="wrong-value cases per run exported for manual labelling")
    ap.add_argument("--out_dir", default="results/error_analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(0)

    table = {}
    for run in args.runs:
        tag = os.path.basename(run.rstrip("/"))
        recs = load_jsonl(os.path.join(run, f"{args.bench}.samples.jsonl"))
        cats, fl, wrong = Counter(), Counter(), []
        for r in recs:
            a = r["samples"][0][args.which]
            c = category(a)
            cats[c] += 1
            if c == "wrong_value":
                gt, _ = parse_gt(r.get("answer"))
                code = extract_code(a["response"])
                ff = flags(gt if gt != NO_SOL else None, a["objective"], code)
                for x in ff:
                    fl[x] += 1
                wrong.append((r, a, gt, ff, code))
        err_types = Counter(r["samples"][0][args.which].get("error_type") for r in recs
                            if r["samples"][0][args.which]["status"] == "runtime_error")
        table[tag] = (len(recs), cats, fl, err_types)
        with open(os.path.join(args.out_dir, f"{tag}.{args.bench}.manual.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "question", "ground_truth", "predicted", "auto_flags", "code"] + MANUAL_COLS)
            for r, a, gt, ff, code in rng.sample(wrong, min(args.manual_n, len(wrong))):
                w.writerow([r["id"], r["question"], r.get("answer"), a["objective"], ";".join(ff), code] + [""] * len(MANUAL_COLS))

    keys = ["correct", "wrong_value", "infeasible", "unbounded", "infeasible_or_unbounded", "runtime_error",
            "syntax_error", "no_code", "no_objective", "timeout"]
    print(f"\n## Error categories on {args.bench} ({args.which} attempt, % of problems)\n")
    print("| run | n | " + " | ".join(keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for tag, (n, cats, fl, et) in table.items():
        print(f"| {tag} | {n} | " + " | ".join(f"{100 * cats.get(k, 0) / n:.1f}" for k in keys) + " |")
    for tag, (n, cats, fl, et) in table.items():
        print(f"\n{tag}: wrong-value heuristic flags {dict(fl)}; runtime error types {dict(et.most_common(8))}")
    print(f"\nmanual-labelling CSVs in {args.out_dir}/")


if __name__ == "__main__":
    main()
