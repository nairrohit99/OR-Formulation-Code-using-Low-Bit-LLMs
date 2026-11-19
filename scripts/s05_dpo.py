"""Step 5: DPO on solver-verified preference pairs, on top of an SFT/RFT LoRA adapter.
The reference policy is the starting adapter; its log-probs are pre-computed once.

python scripts/s05_dpo.py --base ../models/Qwen3-4B-Base --quant nf4 --adapter runs/rft_nf4/final \
    --pairs data/pulp/dpo_nf4.jsonl --out runs/dpo_nf4
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from orlb.data import load_jsonl, split_of  # noqa: E402
from orlb.hf_compat import trainer_kwargs, training_args  # noqa: E402
from orlb.modeling import load_model, load_tokenizer  # noqa: E402
from orlb.training import DPOTrainer, PairCollator, count_trainable, precompute_ref_logps, tokenize_pair  # noqa: E402


def build_pairs(tok, rows, max_len):
    out, dropped = [], 0
    for r in rows:
        c = tokenize_pair(tok, r["prompt"], r["chosen"], max_len)
        j = tokenize_pair(tok, r["prompt"], r["rejected"], max_len)
        if c is None or j is None:
            dropped += 1
            continue
        out.append({"chosen_input_ids": c["input_ids"], "chosen_labels": c["labels"],
                    "rejected_input_ids": j["input_ids"], "rejected_labels": j["labels"]})
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--quant", default="nf4", choices=["nf4", "fp4", "int8", "none"])
    ap.add_argument("--adapter", required=True, help="SFT or RFT adapter = initial policy and reference")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft_weight", type=float, default=1.0, help="weight of NLL on chosen (0 = plain DPO)")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=1, help="pairs per device step")
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_to", default="tensorboard")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = load_tokenizer(args.base, padding_side="right")
    rows = load_jsonl(args.pairs)
    tr_rows = [r for r in rows if split_of(r["id"], args.val_frac, seed=1) == "train"]
    va_rows = [r for r in rows if split_of(r["id"], args.val_frac, seed=1) == "val"]
    train, d1 = build_pairs(tok, tr_rows, args.max_len)
    val, d2 = build_pairs(tok, va_rows, args.max_len)
    print(f"pairs: train {len(train)} val {len(val)} (dropped {d1 + d2} too long)")

    model = load_model(args.base, quant=args.quant, adapter=args.adapter, trainable=True, grad_ckpt=True)
    print(count_trainable(model))
    coll = PairCollator(tok.pad_token_id)
    print("pre-computing reference log-probs ...")
    for ds in (train, val):
        ch, rj = precompute_ref_logps(model, ds, coll)
        for f, a, b in zip(ds, ch, rj):
            f["ref_chosen_logp"], f["ref_rejected_logp"] = a, b
    model.train()

    has_val = len(val) > 0
    targs = training_args(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine", warmup_ratio=0.1,
        max_grad_norm=1.0, bf16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if args.quant != "none" and torch.cuda.is_available() else "adamw_torch",
        logging_steps=5, save_strategy="steps", save_steps=50, save_total_limit=3,
        eval_strategy="steps" if has_val else "no", eval_steps=50, prediction_loss_only=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to, seed=args.seed, remove_unused_columns=False,
    )
    trainer = DPOTrainer(model=model, args=targs, train_dataset=train, eval_dataset=val if has_val else None,
                         data_collator=coll, beta=args.beta, sft_weight=args.sft_weight,
                         **trainer_kwargs(DPOTrainer, tok))
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    print("saved adapter to", final)


if __name__ == "__main__":
    main()
