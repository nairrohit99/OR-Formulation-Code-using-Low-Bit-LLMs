"""Tokenisation with completion-only labels, collators, LoRA config, and a minimal DPO trainer.

Written against plain `transformers.Trainer` + `peft` so it does not depend on the
fast-changing TRL API.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

IGNORE = -100


def tokenize_pair(tok, prompt: str, completion: str, max_len: int) -> Optional[dict]:
    """prompt tokens are masked out of the loss; the EOS token after the completion IS trained."""
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p) + len(c) > max_len:
        return None  # drop instead of truncating: a truncated program teaches broken code
    return {"input_ids": p + c, "labels": [IGNORE] * len(p) + c}


class PadCollator:
    """Right-pads; label padding uses -100 by POSITION, so EOS==PAD does not mask the real EOS."""

    def __init__(self, pad_id: int, keys=("input_ids", "labels")):
        self.pad_id = pad_id
        self.keys = keys

    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats)
        batch = {
            "input_ids": torch.tensor([f["input_ids"] + [self.pad_id] * (L - len(f["input_ids"])) for f in feats]),
            "labels": torch.tensor([f["labels"] + [IGNORE] * (L - len(f["labels"])) for f in feats]),
            "attention_mask": torch.tensor([[1] * len(f["input_ids"]) + [0] * (L - len(f["input_ids"])) for f in feats]),
        }
        return batch


def lora_config(r: int = 64, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


# ----------------------------------------------------------------------------- DPO

class PairCollator:
    """Stacks chosen (first half) and rejected (second half) into one padded batch."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        seqs = [f["chosen_input_ids"] for f in feats] + [f["rejected_input_ids"] for f in feats]
        labs = [f["chosen_labels"] for f in feats] + [f["rejected_labels"] for f in feats]
        L = max(len(s) for s in seqs)
        batch = {
            "input_ids": torch.tensor([s + [self.pad_id] * (L - len(s)) for s in seqs]),
            "labels": torch.tensor([l + [IGNORE] * (L - len(l)) for l in labs]),
            "attention_mask": torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs]),
        }
        if "ref_chosen_logp" in feats[0]:
            batch["ref_chosen_logp"] = torch.tensor([f["ref_chosen_logp"] for f in feats], dtype=torch.float32)
            batch["ref_rejected_logp"] = torch.tensor([f["ref_rejected_logp"] for f in feats], dtype=torch.float32)
        return batch


def sequence_logps(model, input_ids, attention_mask, labels):
    """Sum of log p(label tokens) per sequence, and the number of label tokens."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    tgt = labels[:, 1:]
    mask = tgt != IGNORE
    safe = tgt.masked_fill(~mask, 0)
    # cross_entropy in float32 chunk-free; logits for one pair at 3k tokens fit easily on 80GB
    nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), safe.reshape(-1), reduction="none")
    nll = nll.view(tgt.shape) * mask
    return -nll.sum(-1), mask.sum(-1)


@torch.no_grad()
def precompute_ref_logps(model, dataset, collator, batch_size: int = 1, device=None) -> tuple[list[float], list[float]]:
    model.eval()
    ch, rj = [], []
    for b in range(0, len(dataset), batch_size):
        feats = [dataset[i] for i in range(b, min(b + batch_size, len(dataset)))]
        batch = collator(feats)
        dev = device or next(model.parameters()).device
        lp, _ = sequence_logps(model, batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["labels"].to(dev))
        B = len(feats)
        ch += lp[:B].float().cpu().tolist()
        rj += lp[B:].float().cpu().tolist()
    return ch, rj


class DPOTrainer(Trainer):
    """DPO (Rafailov et al. 2023) + optional length-normalised NLL on the chosen answer
    (as in Iterative Reasoning Preference Optimization, Pang et al. 2024), which keeps the
    likelihood of correct programs from collapsing."""

    def __init__(self, *args, beta: float = 0.1, sft_weight: float = 1.0, **kw):
        super().__init__(*args, **kw)
        self.beta, self.sft_weight = beta, sft_weight
        self._acc = defaultdict(list)
        # our loss is a per-pair mean, so let Trainer divide by gradient_accumulation_steps
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        B = inputs["input_ids"].shape[0] // 2
        lp, ntok = sequence_logps(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
        pc, pr = lp[:B], lp[B:]
        rc = pc.detach() - inputs["ref_chosen_logp"]
        rr = pr.detach() - inputs["ref_rejected_logp"]
        logits = self.beta * ((pc - inputs["ref_chosen_logp"]) - (pr - inputs["ref_rejected_logp"]))
        dpo = -F.logsigmoid(logits).mean()
        nll = -(pc / ntok[:B].clamp(min=1)).mean()
        loss = dpo + self.sft_weight * nll
        tag = "train" if model.training else "eval"
        self._acc[f"{tag}/dpo_loss"].append(dpo.item())
        self._acc[f"{tag}/nll_chosen"].append(nll.item())
        self._acc[f"{tag}/reward_acc"].append((rc > rr).float().mean().item())
        self._acc[f"{tag}/reward_margin"].append((self.beta * (rc - rr)).mean().item())
        return (loss, {"logits": logits}) if return_outputs else loss

    def log(self, logs, *args, **kw):
        tag = "eval" if any(k.startswith("eval") for k in logs) else "train"
        for k in [k for k in self._acc if k.startswith(tag)]:
            v = self._acc.pop(k)
            if v:
                logs[k] = sum(v) / len(v)
        return super().log(logs, *args, **kw)


def count_trainable(model) -> str:
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    return f"trainable {t / 1e6:.1f}M / {a / 1e6:.1f}M params ({100 * t / max(a, 1):.2f}%)"
