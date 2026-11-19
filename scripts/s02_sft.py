"""Step 2 (and step 4 = rejection-sampling fine-tuning): QLoRA / LoRA supervised fine-tuning.

Fixes vs legacy/train.py:
  * loss only on the completion (prompt tokens masked)
  * EOS after the completion is trained (legacy collator masked it because pad == eos)
  * held-out validation loss for checkpoint selection

SFT, 4-bit:   python scripts/s02_sft.py --base ../models/Qwen3-4B-Base --quant nf4  --out runs/sft_nf4
SFT, bf16:    python scripts/s02_sft.py --base ../models/Qwen3-4B-Base --quant none --out runs/sft_bf16
RFT:          python scripts/s02_sft.py --base ../models/Qwen3-4B-Base --quant nf4 \
                  --train data/pulp/sft_train.jsonl data/pulp/rft_nf4.jsonl --out runs/rft_nf4
"""
import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from transformers import Trainer  # noqa: E402

from orlb.data import load_jsonl  # noqa: E402
from orlb.hf_compat import trainer_kwargs, training_args  # noqa: E402
from orlb.modeling import load_model, load_tokenizer  # noqa: E402
from orlb.training import PadCollator, count_trainable, lora_config, tokenize_pair  # noqa: E402


def build(tok, paths, max_len, libs=None):
    rows, dropped = [], 0
    for p in paths:
        for r in load_jsonl(p):
            if libs is not None:
                m = re.search(r"python code using `(\w+)`", r["prompt"])
                libs.add(m.group(1) if m else "?")
            f = tokenize_pair(tok, r["prompt"], r["completion"], max_len)
            if f is None:
                dropped += 1
            else:
                rows.append(f)
    return rows, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--quant", default="nf4", choices=["nf4", "fp4", "int8", "none"])
    ap.add_argument("--train", nargs="+", default=["data/pulp/sft_train.jsonl"])
    ap.add_argument("--val", default="data/pulp/sft_val.jsonl")
    ap.add_argument("--init_adapter", default=None, help="continue training an existing LoRA adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_to", default="tensorboard")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = load_tokenizer(args.base, padding_side="right")
    libs = set()
    train, d1 = build(tok, args.train, args.max_len, libs)
    if len(libs) > 1:
        raise ValueError(f"training files mix solver libraries {libs}; train one adapter per library")
    val, d2 = build(tok, [args.val], args.max_len) if args.val and os.path.exists(args.val) else ([], 0)
    print(f"train {len(train)} (dropped {d1} > max_len), val {len(val)} (dropped {d2})")

    model = load_model(args.base, quant=args.quant, adapter=args.init_adapter, trainable=True, grad_ckpt=True)
    if args.init_adapter is None:
        from peft import get_peft_model
        model = get_peft_model(model, lora_config(args.lora_r, args.lora_alpha, args.lora_dropout))
    print(count_trainable(model))

    has_val = len(val) > 0
    targs = training_args(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.0, max_grad_norm=1.0, bf16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if args.quant != "none" and torch.cuda.is_available() else "adamw_torch",
        logging_steps=10, save_strategy="steps", save_steps=args.eval_steps, save_total_limit=3,
        eval_strategy="steps" if has_val else "no", eval_steps=args.eval_steps,
        load_best_model_at_end=has_val, metric_for_best_model="eval_loss", greater_is_better=False,
        group_by_length=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to, seed=args.seed, remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val if has_val else None,
                      data_collator=PadCollator(tok.pad_token_id), **trainer_kwargs(Trainer, tok))
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    if has_val:
        print("best checkpoint:", trainer.state.best_model_checkpoint, "eval_loss:", trainer.state.best_metric)
    print("saved adapter to", final)


if __name__ == "__main__":
    main()
