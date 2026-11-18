"""Step 4: turn solver-graded samples into
  * an RFT set   (self-generated answers that the solver confirmed correct), and
  * a DPO set    (chosen = correct answer, rejected = wrong/crashing answer, same prompt).

python scripts/s04_build_training_sets.py --samples data/samples_sft_nf4.jsonl \
    --rft_out data/rft_nf4.jsonl --dpo_out data/dpo_nf4.jsonl
"""
import argparse
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orlb.data import load_jsonl, save_jsonl  # noqa: E402
from orlb.execution import extract_code  # noqa: E402


def norm_code(resp):
    code = extract_code(resp) or resp
    code = re.sub(r"#.*", "", code)
    return re.sub(r"\s+", "", code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--rft_out", default="data/rft.jsonl")
    ap.add_argument("--dpo_out", default="data/dpo.jsonl")
    ap.add_argument("--max_rft_per_problem", type=int, default=4)
    ap.add_argument("--max_pairs_per_problem", type=int, default=2)
    ap.add_argument("--include_repairs", action="store_true",
                    help="add successful repair attempts (repair prompt -> fixed answer) to the RFT set")
    ap.add_argument("--gold_chosen", action="store_true",
                    help="when no sample is correct, use the verified gold Pyomo answer as `chosen`")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = [r for p in args.samples for r in load_jsonl(p)]
    rft, dpo, stats = [], [], Counter()
    for r in recs:
        initial = [s["attempts"][0] for s in r["samples"]]
        correct, seen = [], set()
        for a in initial:
            if a["correct"]:
                k = norm_code(a["response"])
                if k not in seen:
                    seen.add(k)
                    correct.append(a)
        wrong = [a for a in initial if not a["correct"]]
        stats["problems"] += 1
        stats["problems_with_correct"] += bool(correct)
        stats["problems_all_correct"] += not wrong

        for a in rng.sample(correct, min(len(correct), args.max_rft_per_problem)):
            rft.append({"id": r["id"], "prompt": r["prompt"], "completion": a["response"].strip(), "kind": "self"})
        if args.include_repairs:
            for s in r["samples"]:
                for a in s["attempts"][1:]:
                    if a["correct"]:
                        rft.append({"id": r["id"], "prompt": a["prompt"], "completion": a["response"].strip(), "kind": "repair"})
                        stats["repair_examples"] += 1

        chosen_pool = [a["response"].strip() for a in correct]
        chosen_kind = "self"
        if not chosen_pool and args.gold_chosen and r.get("gold_completion"):
            chosen_pool, chosen_kind = [r["gold_completion"]], "gold"
        if not chosen_pool or not wrong:
            continue
        # hard negatives first: programs that ran but gave the wrong number
        wrong.sort(key=lambda a: (a["status"] != "ok", rng.random()))
        seen_rej = set()
        for a in wrong:
            if len([d for d in dpo if d["id"] == r["id"]]) >= args.max_pairs_per_problem:
                break
            k = norm_code(a["response"])
            if k in seen_rej:
                continue
            seen_rej.add(k)
            dpo.append({"id": r["id"], "prompt": r["prompt"], "chosen": rng.choice(chosen_pool),
                        "rejected": a["response"].strip(), "chosen_kind": chosen_kind,
                        "rejected_status": "wrong_value" if a["status"] == "ok" else a["status"]})
    save_jsonl(rft, args.rft_out)
    save_jsonl(dpo, args.dpo_out)
    print(dict(stats))
    print(f"RFT examples: {len(rft)} -> {args.rft_out}")
    print(f"DPO pairs: {len(dpo)} -> {args.dpo_out}; rejected types {dict(Counter(d['rejected_status'] for d in dpo))}; "
          f"chosen {dict(Counter(d['chosen_kind'] for d in dpo))}")


if __name__ == "__main__":
    main()
