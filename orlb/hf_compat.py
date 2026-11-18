"""Small shims for differences between transformers versions."""
import inspect

from transformers import TrainingArguments


def training_args(**kw) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in kw and "eval_strategy" not in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    # transformers v5: warmup_ratio -> float warmup_steps; group_by_length -> train_sampling_strategy
    if "warmup_ratio" in kw and "warmup_ratio" not in params:
        kw["warmup_steps"] = float(kw.pop("warmup_ratio"))
    if "group_by_length" in kw and "group_by_length" not in params:
        if kw.pop("group_by_length") and "train_sampling_strategy" in params:
            kw["train_sampling_strategy"] = "group_by_length"
    for old, new in (("gradient_checkpointing", "activation_checkpointing"),
                     ("gradient_checkpointing_kwargs", "activation_checkpointing_kwargs")):
        if old in kw and new in params:
            kw[new] = kw.pop(old)
    dropped = [k for k in list(kw) if k not in params]
    for k in dropped:
        kw.pop(k)
    if dropped:
        print("[hf_compat] TrainingArguments ignores:", dropped)
    return TrainingArguments(**kw)


def trainer_kwargs(trainer_cls, tok) -> dict:
    from transformers import Trainer
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tok}
    return {"tokenizer": tok}
