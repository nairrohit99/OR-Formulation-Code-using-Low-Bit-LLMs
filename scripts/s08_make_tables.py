"""Step 8: paper tables. Accuracy with 95% Wilson CIs, cost columns, and paired McNemar tests
against a baseline run.

python scripts/s08_make_tables.py --runs results/* --baseline results/sft_nf4_greedy --metric first_sample_acc
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orlb.data import load_jsonl  # noqa: E402
from orlb.grading import mcnemar_exact, wilson_ci  # noqa: E402


def per_problem(run, bench, which="final"):
    p = os.path.join(run, f"{bench}.samples.jsonl")
    if not os.path.exists(p):
        return None
    return {r["id"]: r["samples"][0][which]["correct"] for r in load_jsonl(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--metric", default="first_sample_acc",
                    help="first_sample_acc | mean@1 | maj@8 | pass@8 | ..._lenient")
    ap.add_argument("--stage", default="after_repair", choices=["before_repair", "after_repair"])
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    runs = [r for pat in args.runs for r in glob.glob(pat) if os.path.exists(os.path.join(r, "summary.json"))]
    data = {r: json.load(open(os.path.join(r, "summary.json"))) for r in sorted(set(runs))}
    benches = sorted({b for s in data.values() for b in s["benchmarks"]})

    sep = " & " if args.latex else " | "
    end = r" \\" if args.latex else " |"
    head = ["run", "lib", "quant", "k", "rep"] + benches + ["tok/prob", "s/prob", "peak GB"]
    print(("" if args.latex else "| ") + sep.join(head) + end)
    if not args.latex:
        print("|" + "---|" * len(head))
    for run, s in data.items():
        c = s["config"]
        cells = [os.path.basename(run), c.get("library", "pyomo"), c["quant"], str(c["n"]), str(c["repair_rounds"])]
        toks, secs, mem = [], [], []
        for b in benches:
            if b not in s["benchmarks"]:
                cells.append("-")
                continue
            bs = s["benchmarks"][b]
            st = bs[args.stage]
            v = st.get(args.metric)
            if v is None:
                cells.append("-")
            else:
                n = st["n_problems"]
                lo, hi = wilson_ci(round(v * n), n)
                cells.append(f"{100 * v:.1f} [{100 * lo:.0f},{100 * hi:.0f}]")
            pp = bs["cost"]["per_problem"]
            toks.append(pp["tokens"]), secs.append(pp["gpu_seconds"]), mem.append(pp["peak_mem_gb"])
        cells += [f"{sum(toks) / max(1, len(toks)):.0f}", f"{sum(secs) / max(1, len(secs)):.2f}", f"{max(mem or [0]):.1f}"]
        print(("" if args.latex else "| ") + sep.join(cells) + end)

    if args.baseline:
        print(f"\nPaired McNemar vs {os.path.basename(args.baseline)} (first sample, {args.stage}):")
        which = "final" if args.stage == "after_repair" else "first"
        for b in benches:
            base = per_problem(args.baseline, b, which)
            if base is None:
                continue
            for run in data:
                if os.path.abspath(run) == os.path.abspath(args.baseline):
                    continue
                other = per_problem(run, b, which)
                if other is None:
                    continue
                ids = sorted(set(base) & set(other))
                n01, n10, p = mcnemar_exact([base[i] for i in ids], [other[i] for i in ids])
                print(f"  {b:12s} {os.path.basename(run):30s} base-only {n01:3d}  run-only {n10:3d}  p={p:.3g}")


if __name__ == "__main__":
    main()
