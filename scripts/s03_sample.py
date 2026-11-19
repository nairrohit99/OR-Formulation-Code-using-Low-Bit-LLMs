"""Step 3: sample k answers per TRAINING problem with the fine-tuned model, run them, and grade
them against the coptpy reference objective. Resumable (appends chunk by chunk).

The library (PuLP / Pyomo) is taken from the data file written by s01_translate.py.

python scripts/s03_sample.py --base ../models/Qwen3-4B-Base --quant nf4 --adapter runs/sft_nf4/final \
    --data data/pulp/or_instruct.jsonl --n 8 --temperature 0.8 --repair_rounds 1 --out data/pulp/samples_sft_nf4.jsonl
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orlb.data import append_jsonl, load_jsonl  # noqa: E402
from orlb.modeling import Generator, load_model, load_tokenizer  # noqa: E402
from orlb.pipeline import solve_items  # noqa: E402
from orlb.prompts import check_library, task_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/pulp/or_instruct.jsonl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repair_rounds", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64, help="sequences per generate() call")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # skip problems the translator flagged as outside the library's model class (e.g. quadratic for PuLP)
    items = [it for it in load_jsonl(args.data)
             if it["split"] == args.split and it.get("answer") is not None and not it.get("refused")]
    libs = {it.get("library", "pulp") for it in items}
    if len(libs) != 1:
        raise ValueError(f"data file mixes libraries {libs}")
    library = libs.pop()
    check_library(library)
    items = items[: args.limit]
    done = {r["id"] for r in load_jsonl(args.out)} if os.path.exists(args.out) else set()
    items = [it for it in items if it["id"] not in done]
    print(f"{len(items)} problems to sample with {library} ({len(done)} already done)")
    if not items:
        return
    tok = load_tokenizer(args.base)
    model = load_model(args.base, quant=args.quant, adapter=args.adapter)
    gen = Generator(model, tok, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    keep = ("id", "question", "answer", "completion")
    for c in range(0, len(items), args.chunk):
        chunk = [{k: it.get(k) for k in keep} for it in items[c:c + args.chunk]]
        prompts = [task_prompt(it["question"], library) for it in chunk]
        recs, stats = solve_items(gen, chunk, prompts, n=args.n, temperature=args.temperature, top_p=args.top_p,
                                  seed=args.seed + c, repair_rounds=args.repair_rounds)
        for r in recs:
            r["gold_completion"] = r.pop("completion")
            r["prompt"] = task_prompt(r["question"], library)
            r["library"] = library
            append_jsonl(r, args.out)
        first = Counter("correct" if s["first"]["correct"] else s["first"]["status"] for r in recs for s in r["samples"])
        solved = sum(any(s["final"]["correct"] for s in r["samples"]) for r in recs)
        print(f"[{c + len(chunk)}/{len(items)}] solved(any) {solved}/{len(recs)}  first-attempt {dict(first)}  "
              f"gen {stats['generation']['seconds']:.0f}s repair {stats['repair']['seconds']:.0f}s")


if __name__ == "__main__":
    main()
