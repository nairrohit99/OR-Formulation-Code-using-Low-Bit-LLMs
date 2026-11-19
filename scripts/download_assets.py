"""Run this on a machine WITH internet, then copy the folders to the offline A100 machine.

python scripts/download_assets.py --models_dir ../models --datasets_dir ../datasets
Also run:  pip download -r requirements.txt -d ../wheels   (then offline: pip install --no-index --find-links ../wheels -r requirements.txt)
"""
import argparse
import os

MODELS = [
    "Qwen/Qwen3-4B-Base",            # main model (quantised on the fly -> same weights for nf4 and bf16)
    "Qwen/Qwen3-8B-Base",            # bigger full-precision comparison (budget-matched experiment)
    "Qwen/Qwen3-1.7B-Base",          # optional: size axis
    "Qwen/Qwen3-32B",                # translator for coptpy -> PuLP / Pyomo (run in nf4); gpt-oss-20b also works
]
DATASETS = [
    "CardinalOperations/OR-Instruct-Data-3K",
    "CardinalOperations/NL4OPT",
    "CardinalOperations/MAMO",
    "CardinalOperations/IndustryOR",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="../models")
    ap.add_argument("--datasets_dir", default="../datasets")
    ap.add_argument("--skip_models", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    todo = [] if args.skip_models else [(m, "model", args.models_dir) for m in MODELS]
    todo += [(d, "dataset", args.datasets_dir) for d in DATASETS]
    for repo, kind, root in todo:
        dst = os.path.join(root, repo.split("/")[-1])
        try:
            snapshot_download(repo_id=repo, repo_type=kind, local_dir=dst)
            print("ok  ", repo, "->", dst)
        except Exception as e:  # names of benchmark repos change; fall back to the ORLM / LLMOPT GitHub repos
            print("FAIL", repo, type(e).__name__, str(e)[:120])
    print("\nBenchmarks: any .jsonl/.json/.parquet with a question field (question/en_question) and an answer "
          "field (answer/en_answer) works with scripts/s06_evaluate.py.")


if __name__ == "__main__":
    main()
