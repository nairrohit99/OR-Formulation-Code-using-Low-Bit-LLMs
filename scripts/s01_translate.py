"""Step 1: translate the coptpy solutions of OR-Instruct-Data-3K into the target library
(PuLP by default, or Pyomo) with a larger local LLM. A translation is kept only if it
reproduces the coptpy objective value (solver-verified).

Output (in data/<library>/):
  or_instruct.jsonl          every problem: ref answer, translated completion (or null), split
  sft_train.jsonl, sft_val.jsonl   verified (prompt, completion) pairs for SFT

python scripts/s01_translate.py --library pulp --ref data/or_instruct_ref.jsonl \
    --translator ../models/Qwen3-32B --quant nf4 --rounds 4
"""
import argparse
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orlb.data import build_completion, load_jsonl, math_part, save_jsonl, split_of  # noqa: E402
from orlb.execution import run_many  # noqa: E402
from orlb.grading import NO_SOL, match, pred_value  # noqa: E402
from orlb.modeling import Generator, load_model, load_tokenizer  # noqa: E402
from orlb.prompts import DEFAULT_LIBRARY, LIBRARIES, check_library, get_library, task_prompt, translate_prompt  # noqa: E402

_BLOCK = re.compile(r"```[ \t]*(?:python|py)[ \t]*\n(.*?)```", re.S | re.I)
_REFUSALS = ("NOT_LINEAR", "NOT_SUPPORTED")


def last_code_block(text):
    blocks = _BLOCK.findall(text or "")
    return blocks[-1] if blocks else None


def refused(text):
    tail = (text or "").strip()[-200:]
    return any(r in tail for r in _REFUSALS) and last_code_block(text) is None


def looks_legit(code, lib):
    return bool(code) and "OBJECTIVE_VALUE" in code and all(m in code for m in lib.required_markers)


def chat(tok, content):
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": content}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return content + "\n\n"


def ref_answer(it):
    if it["ref_status"] == "ok":
        return it["ref_objective"]
    if it["ref_status"] in ("infeasible", "unbounded", "infeasible_or_unbounded"):
        return NO_SOL
    return None


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

    lib = get_library(args.library)
    check_library(lib.name)
    out_dir = args.out_dir or os.path.join("data", lib.name)
    items = load_jsonl(args.ref)[: args.limit]
    for it in items:
        it["answer"] = ref_answer(it)
        it["split"] = split_of(it["id"], args.val_frac)
        it.update(library=lib.name, target_code=None, target_objective=None, verified_by=None,
                  refused=False, attempt_log=[])
    tok = load_tokenizer(args.translator)
    model = load_model(args.translator, quant=args.quant)
    gen = Generator(model, tok, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, use_stop_strings=False)

    for rnd in range(args.rounds):
        todo = [it for it in items if it["target_code"] is None and not it["refused"] and it.get("orig_code")
                and (it["answer"] is not None or args.allow_agreement)]
        print(f"round {rnd + 1}: {len(todo)} problems to translate to {lib.name}", flush=True)
        if not todo:
            break
        prompts = [chat(tok, translate_prompt(it["question"], it["orig_code"], lib.name)) for it in todo]
        temp = 0.0 if rnd == 0 else args.temperature
        outs, st = gen.generate(prompts, n=1, temperature=temp, top_p=0.95, seed=rnd)
        codes = [last_code_block(o[0]) for o in outs]
        res = run_many([c if looks_legit(c, lib) else None for c in codes], timeout=60)
        for it, out, code, r in zip(todo, outs, codes, res):
            if refused(out[0]):
                it["refused"] = True  # translator says the model is outside what the library supports
                it["attempt_log"].append({"status": "refused"})
                continue
            pv = pred_value(r.status, r.objective)
            it["attempt_log"].append({"status": r.status, "objective": r.objective, "legit": looks_legit(code, lib)})
            if it["answer"] is not None:
                if match(pv, it["answer"], rel_tol=args.rel_tol):
                    it["target_code"], it["target_objective"], it["verified_by"] = code, r.objective, "reference"
            elif pv is not None:  # agreement mode
                prev = [a for a in it["attempt_log"][:-1] if pred_value(a.get("status"), a.get("objective")) is not None]
                if any(match(pred_value(a["status"], a["objective"]), pv, rel_tol=args.rel_tol) for a in prev):
                    it["target_code"], it["target_objective"], it["verified_by"] = code, r.objective, "agreement"
                    it["answer"] = pv
        print(f"  verified so far: {sum(it['target_code'] is not None for it in items)}  "
              f"refused: {sum(it['refused'] for it in items)}  ({st.seconds:.0f}s)", flush=True)

    sft = {"train": [], "val": []}
    for it in items:
        it["prompt"] = task_prompt(it["question"], lib.name)
        it["completion"] = (build_completion(math_part(it["orig_completion"], lib.name), it["target_code"], lib.name)
                            if it["target_code"] else None)
        if it["completion"]:
            sft[it["split"]].append({"id": it["id"], "prompt": it["prompt"], "completion": it["completion"]})
    save_jsonl(items, os.path.join(out_dir, "or_instruct.jsonl"))
    for k, v in sft.items():
        save_jsonl(v, os.path.join(out_dir, f"sft_{k}.jsonl"))
    print("verified_by:", Counter(it["verified_by"] for it in items))
    print(f"refused by translator (outside {lib.name}'s model classes): {sum(it['refused'] for it in items)}")
    print("ref status of unverified:", Counter(it["ref_status"] for it in items if not it["target_code"]))
    print(f"SFT train {len(sft['train'])}, val {len(sft['val'])} -> {out_dir}/")


if __name__ == "__main__":
    main()
