"""Step 6: evaluate a model on OR benchmarks with solver execution.

Measures accuracy (strict and rounding-lenient), maj@k / pass@k, the effect of repair rounds,
and the cost: peak GPU memory, generated tokens, wall-clock time.

Greedy:      python scripts/s06_evaluate.py --base ../models/Qwen3-4B-Base --quant nf4 --adapter runs/dpo_nf4/final \
                 --library pulp --bench nl4opt=data/nl4opt_test.jsonl mamo_easy=data/mamo_easy.jsonl --tag dpo_nf4_greedy
Voting:      ... --n 8 --temperature 0.7 --tag dpo_nf4_k8
Repair:      ... --repair_rounds 2 --tag dpo_nf4_greedy_rep2
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from orlb.data import load_benchmark, load_jsonl, save_jsonl  # noqa: E402
from orlb.modeling import Generator, load_model, load_tokenizer  # noqa: E402
from orlb.pipeline import score_records, solve_items  # noqa: E402
from orlb.prompts import DEFAULT_LIBRARY, LIBRARIES, check_library, task_prompt  # noqa: E402


def fewshot_prefix(path, k, library):
    if not path or k <= 0:
        return ""
    rows = load_jsonl(path)
    if rows and not all(f"`{library}`" in r["prompt"] for r in rows[:k]):
        raise ValueError(f"few-shot file {path} was not built for library {library}")
    rows.sort(key=lambda r: len(r["prompt"]) + len(r["completion"]))  # shortest examples keep context small
    return "".join(r["prompt"] + r["completion"] + "\n\n" for r in rows[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--quant", default="nf4", choices=["nf4", "fp4", "int8", "none"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--library", default=DEFAULT_LIBRARY, choices=sorted(LIBRARIES),
                    help="must match the library the adapter was trained on")
    ap.add_argument("--bench", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repair_rounds", type=int, default=0)
    ap.add_argument("--repair_no_solution", action="store_true", help="also repair answers reported infeasible/unbounded")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--fewshot_file", default=None, help="e.g. data/pulp/sft_train.jsonl (for the un-tuned base model)")
    ap.add_argument("--fewshot_k", type=int, default=0)
    ap.add_argument("--rel_tol", type=float, default=1e-4)
    ap.add_argument("--exec_timeout", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    check_library(args.library)
    out = os.path.join(args.out_dir, args.tag)
    os.makedirs(out, exist_ok=True)
    tok = load_tokenizer(args.base)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = load_model(args.base, quant=args.quant, adapter=args.adapter)
    load_s = time.time() - t0
    weights_gb = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else None
    gen = Generator(model, tok, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    prefix = fewshot_prefix(args.fewshot_file, args.fewshot_k, args.library)

    all_summary = {"config": vars(args), "model_weights_gb": weights_gb, "load_seconds": load_s, "benchmarks": {}}
    for spec in args.bench:
        name, path = spec.split("=", 1)
        items = load_benchmark(path, name)[: args.limit]
        prompts = [prefix + task_prompt(it["question"], args.library) for it in items]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        recs, cost = solve_items(gen, items, prompts, n=args.n, temperature=args.temperature, top_p=args.top_p,
                                 seed=args.seed, repair_rounds=args.repair_rounds,
                                 repair_no_solution=args.repair_no_solution, exec_timeout=args.exec_timeout,
                                 rel_tol=args.rel_tol)
        summ = {
            "before_repair": score_records(recs, rel_tol=args.rel_tol, which="first"),
            "after_repair": score_records(recs, rel_tol=args.rel_tol, which="final"),
            "cost": cost,
        }
        g, r = cost["generation"], cost["repair"]
        n = max(1, len(items))
        summ["cost"]["per_problem"] = {
            "tokens": (g["new_tokens"] + r["new_tokens"]) / n,
            "gpu_seconds": (g["seconds"] + r["seconds"]) / n,
            "peak_mem_gb": max(g["peak_mem_gb"], r["peak_mem_gb"]),
        }
        all_summary["benchmarks"][name] = summ
        save_jsonl(recs, os.path.join(out, f"{name}.samples.jsonl"))
        b, a = summ["before_repair"], summ["after_repair"]
        print(f"[{name}] n={len(items)}  first-sample acc {b['first_sample_acc']:.3f} "
              f"(lenient {b['first_sample_acc_lenient']:.3f})  after repair {a['first_sample_acc']:.3f}  "
              + (f"maj@{args.n} {a.get(f'maj@{args.n}', float('nan')):.3f}  pass@{args.n} {a.get(f'pass@{args.n}', float('nan')):.3f}  " if args.n > 1 else "")
              + f"| {summ['cost']['per_problem']['gpu_seconds']:.2f}s/problem, peak {summ['cost']['per_problem']['peak_mem_gb']:.1f}GB")
        print("   breakdown:", b["breakdown_first_sample"])
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(all_summary, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
