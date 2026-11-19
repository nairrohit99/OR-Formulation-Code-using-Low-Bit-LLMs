"""Contamination check: word n-gram overlap between training questions and benchmark questions.
Reviewers will ask; report the number of near-duplicates (and optionally exclude them).

python scripts/check_overlap.py --train data/pulp/or_instruct.jsonl --bench nl4opt=data/nl4opt_test.jsonl
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orlb.data import load_benchmark, load_jsonl  # noqa: E402


def grams(text, n):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--bench", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5, help="fraction of benchmark n-grams found in one training question")
    args = ap.parse_args()

    train = [(r["id"], grams(r["question"], args.n)) for r in load_jsonl(args.train)]
    index = {}
    for i, (_, g) in enumerate(train):
        for x in g:
            index.setdefault(x, []).append(i)
    for spec in args.bench:
        name, path = spec.split("=", 1)
        flagged = []
        for it in load_benchmark(path, name):
            g = grams(it["question"], args.n)
            if not g:
                continue
            hits = {}
            for x in g:
                for i in index.get(x, ()):
                    hits[i] = hits.get(i, 0) + 1
            if hits:
                i, h = max(hits.items(), key=lambda kv: kv[1])
                if h / len(g) >= args.threshold:
                    flagged.append((it["id"], train[i][0], h / len(g)))
        print(f"{name}: {len(flagged)} benchmark questions with >= {args.threshold:.0%} {args.n}-gram overlap")
        for f in flagged[:10]:
            print("   ", f)


if __name__ == "__main__":
    main()
