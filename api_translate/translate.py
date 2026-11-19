"""Translate the coptpy solutions of OR-Instruct-Data-3K into Pyomo (or PuLP) with the Claude or
Gemini API, verify every translation with the solver, retry failures with feedback, and write
training files that the rest of the pipeline uses unchanged.

Run on a machine WITH internet that has pyomo + highspy (or pulp==2.9.0) installed; no GPU needed.
Then copy data/<library>/ to the offline A100 machine.

  # Claude (default model claude-opus-5-5)
  python api_translate/translate.py --provider claude --library pyomo --ref data/or_instruct_ref.jsonl
  # Claude via Message Batches (50% cheaper, results usually within an hour per round)
  python api_translate/translate.py --provider claude --batch --library pyomo
  # Gemini (pass the exact model id)
  python api_translate/translate.py --provider gemini --model <gemini-model-id> --library pyomo
  # See the prompts and request count without calling any API
  python api_translate/translate.py --provider claude --dry_run --limit 3

Every API response is cached in api_translate/cache/, so a crashed or interrupted run can be
restarted with the same command without paying for the same requests twice.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orlb.data import load_jsonl  # noqa: E402
from orlb.execution import run_many  # noqa: E402
from orlb.prompts import LIBRARIES, check_library  # noqa: E402
from orlb.translation import (feedback_prompt, init_items, last_code_block, looks_legit,  # noqa: E402
                              needs_translation, record_attempt, write_outputs)
from providers import Completion, make_provider  # noqa: E402


class Cache:
    """Append-only JSONL cache: request key -> completion. Thread-safe."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.data = {}
        if os.path.exists(path):
            for r in load_jsonl(path):
                self.data[r["key"]] = r
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def get(self, key):
        r = self.data.get(key)
        return None if r is None else Completion(text=r["text"], ok=r["ok"], error=r.get("error"),
                                                  input_tokens=r.get("input_tokens", 0),
                                                  output_tokens=r.get("output_tokens", 0), meta=r.get("meta", {}))

    def put(self, key, item_id, c: Completion):
        r = {"key": key, "id": item_id, "text": c.text, "ok": c.ok, "error": c.error, "input_tokens": c.input_tokens,
             "output_tokens": c.output_tokens, "meta": c.meta, "time": time.time()}
        with self.lock:
            self.data[key] = r
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def request_key(provider_tag: str, prompt: str) -> str:
    return hashlib.sha256((provider_tag + "\x00" + prompt).encode()).hexdigest()[:40]


def run_sync(provider, cache, jobs, workers):
    """jobs: list of (key, item_id, prompt). Calls the API for uncached jobs in parallel."""
    todo = [j for j in jobs if cache.get(j[0]) is None]
    done = [0]

    def call(job):
        key, item_id, prompt = job
        c = provider.complete(prompt)
        if c.ok or c.error == "refusal":   # transient errors are not cached, so a rerun retries them
            cache.put(key, item_id, c)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  {done[0]}/{len(todo)} API calls done", flush=True)
        return key, c

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fresh = dict(ex.map(call, todo))
    return {k: (cache.get(k) or fresh.get(k)) for k, _, _ in jobs}


def run_batch(provider, cache, jobs, state_path, poll_seconds):
    """Claude Message Batches. The batch id is saved, so an interrupted run resumes polling."""
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    for batch_id in list(state):             # finish batches from an interrupted run first
        print(f"resuming batch {batch_id}", flush=True)
        for cid, c in provider.wait_batch(batch_id, poll_seconds).items():
            key, item_id = state[batch_id][cid]
            if c.ok or c.error == "refusal":
                cache.put(key, item_id, c)
        del state[batch_id]
        json.dump(state, open(state_path, "w"))
    todo = [j for j in jobs if cache.get(j[0]) is None]
    if todo:
        reqs, mapping = [], {}
        for i, (key, item_id, prompt) in enumerate(todo):
            cid = f"r{i}_{key[:24]}"
            reqs.append((cid, prompt))
            mapping[cid] = (key, item_id)
        batch_id = provider.submit_batch(reqs)
        state[batch_id] = mapping
        json.dump(state, open(state_path, "w"))
        print(f"submitted batch {batch_id} with {len(reqs)} requests", flush=True)
        for cid, c in provider.wait_batch(batch_id, poll_seconds).items():
            key, item_id = mapping[cid]
            if c.ok or c.error == "refusal":
                cache.put(key, item_id, c)
        del state[batch_id]
        json.dump(state, open(state_path, "w"))
    return {k: (cache.get(k) or Completion(ok=False, error="batch_failed")) for k, _, _ in jobs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["claude", "gemini"], required=True)
    ap.add_argument("--model", default=None, help="Claude default: claude-opus-5-5. Gemini: required")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude only. Raise to `high` if many problems stay unverified")
    ap.add_argument("--no_fallbacks", action="store_true", help="Claude: disable the server-side refusal fallback")
    ap.add_argument("--temperature", type=float, default=None, help="Gemini only")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--library", default="pyomo", choices=sorted(LIBRARIES))
    ap.add_argument("--ref", default="data/or_instruct_ref.jsonl", help="output of scripts/s00_reference_answers.py")
    ap.add_argument("--rounds", type=int, default=3, help="attempts per problem (round 2+ include solver feedback)")
    ap.add_argument("--workers", type=int, default=8, help="parallel API calls (sync mode)")
    ap.add_argument("--batch", action="store_true", help="Claude: use Message Batches (50%% cheaper, slower)")
    ap.add_argument("--poll_seconds", type=int, default=60)
    ap.add_argument("--exec_workers", type=int, default=None)
    ap.add_argument("--rel_tol", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--allow_agreement", action="store_true")
    ap.add_argument("--out_dir", default=None, help="default: data/<library>")
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "api_translate", "cache"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true", help="print the first prompt and request count; no API calls")
    args = ap.parse_args()

    if args.batch and args.provider != "claude":
        raise SystemExit("--batch is only implemented for Claude")
    check_library(args.library)
    items = init_items(load_jsonl(args.ref)[: args.limit], args.library, args.val_frac)
    print(f"{len(items)} problems; {sum(needs_translation(it, args.allow_agreement) for it in items)} translatable "
          f"(have coptpy code and a reference answer)")
    if args.dry_run:
        first = next(it for it in items if needs_translation(it, args.allow_agreement))
        print("=" * 80 + "\n" + feedback_prompt(first, args.library) + "\n" + "=" * 80)
        print(f"round 1 would send {sum(needs_translation(it, args.allow_agreement) for it in items)} requests; "
              f"later rounds only for failures (max {args.rounds} rounds)")
        return

    provider = make_provider(args.provider, args.model, args.effort, args.max_tokens, not args.no_fallbacks,
                             args.temperature)
    tag = f"{args.provider}|{provider.model}|{args.effort if args.provider == 'claude' else args.temperature}|{args.max_tokens}"
    safe = provider.model.replace("/", "_")
    cache = Cache(os.path.join(args.cache_dir, f"{args.provider}_{safe}_{args.library}.jsonl"))
    state_path = os.path.join(args.cache_dir, f"batches_{safe}_{args.library}.json")
    print(f"provider {tag}; cache {cache.path} ({len(cache.data)} cached responses)")

    tokens_in = tokens_out = 0
    for rnd in range(args.rounds):
        todo = [it for it in items if needs_translation(it, args.allow_agreement)]
        print(f"\nround {rnd + 1}: {len(todo)} problems to translate to {args.library}", flush=True)
        if not todo:
            break
        jobs = []
        for it in todo:
            prompt = feedback_prompt(it, args.library)
            jobs.append((request_key(tag, prompt), it["id"], prompt))
        cached_before = sum(cache.get(k) is not None for k, _, _ in jobs)
        t0 = time.time()
        if args.batch:
            comps = run_batch(provider, cache, jobs, state_path, args.poll_seconds)
        else:
            comps = run_sync(provider, cache, jobs, args.workers)
        texts = [comps[k].text if comps[k] and comps[k].ok else "" for k, _, _ in jobs]
        errors = [comps[k].error for k, _, _ in jobs if comps[k] and not comps[k].ok]
        for k, _, _ in jobs:
            if comps[k]:
                tokens_in += comps[k].input_tokens
                tokens_out += comps[k].output_tokens
        codes = [last_code_block(t) for t in texts]
        res = run_many([c if looks_legit(c, args.library) else None for c in codes], timeout=60, workers=args.exec_workers)
        for it, text, code, r, (k, _, _) in zip(todo, texts, codes, res, jobs):
            if comps[k] is not None and not comps[k].ok and comps[k].error != "refusal":
                continue  # transient API failure: not an attempt, retried next round / next run
            record_attempt(it, text, code, r, args.library, args.rel_tol, args.allow_agreement)
        print(f"  {cached_before} from cache, {len(jobs) - cached_before} new calls, {len(errors)} API errors "
              f"({time.time() - t0:.0f}s)")
        if errors:
            print("  e.g.", errors[:3])
        print(f"  verified so far: {sum(it['target_code'] is not None for it in items)}/{len(items)}  "
              f"refused (outside {args.library}): {sum(it['refused'] for it in items)}", flush=True)

    out_dir = args.out_dir or os.path.join(ROOT, "data", args.library)
    for it in items:
        it["translator"] = tag
    stats = write_outputs(items, args.library, out_dir)
    print("\n", stats)
    print(f"tokens (this run, incl. cached): input {tokens_in:,}  output {tokens_out:,}")
    print(f"saved to {out_dir}/  ->  train with: python scripts/s02_sft.py --base <model> "
          f"--train {out_dir}/sft_train.jsonl --val {out_dir}/sft_val.jsonl --out runs/sft_{args.library}")


if __name__ == "__main__":
    main()
