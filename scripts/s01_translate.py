"""Step 1: translate the coptpy solutions of OR-Instruct-Data-3K into the target library
(PuLP by default, or Pyomo) with a larger LOCAL LLM. A translation is kept only if it
reproduces the coptpy objective value (solver-verified). Failed attempts are retried with
the execution feedback added to the prompt.

To use the Claude or Gemini APIs instead (on a machine with internet), see api_translate/.

Output (in data/<library>/):
  or_instruct.jsonl          every problem: ref answer, translated completion (or null), split
  sft_train.jsonl, sft_val.jsonl   verified (prompt, completion) pairs for SFT

python scripts/s01_translate.py --library pulp --ref data/or_instruct_ref.jsonl \
    --translator ../models/Qwen3-32B --quant nf4 --rounds 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orlb.data import load_jsonl  # noqa: E402
from orlb.execution import run_many  # noqa: E402
from orlb.modeling import Generator, load_model, load_tokenizer  # noqa: E402
from orlb.prompts import DEFAULT_LIBRARY, LIBRARIES, check_library  # noqa: E402
from orlb.translation import (feedback_prompt, init_items, last_code_block, looks_legit,  # noqa: E402
                              needs_translation, record_attempt, write_outputs)


def chat(tok, content):
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": content}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return content + "\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=DEFAULT_LIBRARY, choices=sorted(LIBRARIES))
    ap.add_argument("--ref", default="data/or_instruct_ref.jsonl")
    ap.add_argument("--translator", required=True, help="local path of an instruction-tuned LLM (e.g. Qwen3-32B, gpt-oss-20b)")
    ap.add_argument("--quant", default="nf4", choices=["nf4", "int8", "none"])
    ap.add_argument("--rounds", type=int, default=4, help="max translation attempts per problem")
    ap.add_argument("--temperature", type=float, default=0.6, help="for rounds after the first (first is greedy)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--rel_tol", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--allow_agreement", action="store_true",
                    help="for problems whose coptpy reference failed, accept a translation if 2 attempts agree")
    ap.add_argument("--out_dir", default=None, help="default: data/<library>")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    check_library(args.library)
    out_dir = args.out_dir or os.path.join("data", args.library)
    items = init_items(load_jsonl(args.ref)[: args.limit], args.library, args.val_frac)
    tok = load_tokenizer(args.translator)
    model = load_model(args.translator, quant=args.quant)
    gen = Generator(model, tok, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, use_stop_strings=False)

    for rnd in range(args.rounds):
        todo = [it for it in items if needs_translation(it, args.allow_agreement)]
        print(f"round {rnd + 1}: {len(todo)} problems to translate to {args.library}", flush=True)
        if not todo:
            break
        prompts = [chat(tok, feedback_prompt(it, args.library)) for it in todo]
        outs, st = gen.generate(prompts, n=1, temperature=0.0 if rnd == 0 else args.temperature, top_p=0.95, seed=rnd)
        codes = [last_code_block(o[0]) for o in outs]
        res = run_many([c if looks_legit(c, args.library) else None for c in codes], timeout=60)
        for it, out, code, r in zip(todo, outs, codes, res):
            record_attempt(it, out[0], code, r, args.library, args.rel_tol, args.allow_agreement)
        print(f"  verified so far: {sum(it['target_code'] is not None for it in items)}  "
              f"refused: {sum(it['refused'] for it in items)}  ({st.seconds:.0f}s)", flush=True)

    stats = write_outputs(items, args.library, out_dir)
    print(stats)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
