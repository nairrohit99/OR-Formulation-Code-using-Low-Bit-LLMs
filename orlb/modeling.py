"""Model / tokenizer loading and batched generation (plain HF transformers, no vLLM)."""
from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .prompts import STOP_STRINGS

try:
    import transformers
    from packaging.version import Version
    _DTYPE_KW = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
except Exception:
    _DTYPE_KW = "torch_dtype"


def is_prequantized(path: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    except Exception:
        return False
    return getattr(cfg, "quantization_config", None) is not None


def bnb_config(quant: str):
    if quant in ("none", "bf16", None):
        return None
    from transformers import BitsAndBytesConfig
    if quant == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16)
    if quant == "fp4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.bfloat16)
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"unknown quant {quant}")


def load_tokenizer(path: str, padding_side: str = "left"):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_model(base: str, quant: str = "nf4", adapter: Optional[str] = None, trainable: bool = False,
               grad_ckpt: bool = False):
    """quant: nf4 | fp4 | int8 | none. A pre-quantized checkpoint keeps its own quantization."""
    kw = {_DTYPE_KW: torch.bfloat16, "trust_remote_code": True}
    if torch.cuda.is_available():
        kw["device_map"] = {"": torch.cuda.current_device()}
    if is_prequantized(base):
        if quant in ("none", "bf16"):
            warnings.warn(f"{base} is pre-quantized; quant='{quant}' ignored.")
    elif bnb_config(quant) is not None:
        kw["quantization_config"] = bnb_config(quant)
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    quantized = bool(getattr(model, "is_quantized", False) or getattr(model, "is_loaded_in_4bit", False)
                     or getattr(model, "is_loaded_in_8bit", False))
    if trainable:
        model.config.use_cache = False
        if quantized:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=grad_ckpt,
                gradient_checkpointing_kwargs={"use_reentrant": False})
        elif grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    if not trainable:
        model.eval()
    return model


def model_device(model):
    return next(model.parameters()).device


@dataclass
class GenStats:
    sequences: int = 0
    new_tokens: int = 0
    seconds: float = 0.0
    peak_mem_gb: float = 0.0

    def add(self, other: "GenStats"):
        self.sequences += other.sequences
        self.new_tokens += other.new_tokens
        self.seconds += other.seconds
        self.peak_mem_gb = max(self.peak_mem_gb, other.peak_mem_gb)

    def to_dict(self):
        return asdict(self)


def truncate_at_stop(text: str, stops=STOP_STRINGS) -> str:
    cut = len(text)
    for s in stops:
        i = text.find(s, 1)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


class Generator:
    """Batched sampling with HF `generate`. Batches are length-sorted to reduce padding."""

    def __init__(self, model, tok, max_new_tokens: int = 2048, batch_size: int = 32, top_k: int = 0,
                 use_stop_strings: bool = True):
        self.model, self.tok = model, tok
        self.max_new_tokens, self.batch_size, self.top_k = max_new_tokens, batch_size, top_k
        self.use_stop_strings = use_stop_strings
        eos = {tok.eos_token_id}
        gc_eos = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(gc_eos, int):
            eos.add(gc_eos)
        elif isinstance(gc_eos, (list, tuple)):
            eos.update(gc_eos)
        self.eos_ids = sorted(e for e in eos if e is not None)

    @torch.no_grad()
    def generate(self, prompts: list[str], n: int = 1, temperature: float = 0.0, top_p: float = 1.0,
                 seed: int = 0, progress: bool = True) -> tuple[list[list[str]], GenStats]:
        if temperature <= 0 and n > 1:
            raise ValueError("greedy decoding with n>1 gives identical samples; set temperature>0")
        dev = model_device(self.model)
        stats = GenStats()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        per_batch = max(1, self.batch_size // n)
        out: list[Optional[list[str]]] = [None] * len(prompts)
        batches = [order[b:b + per_batch] for b in range(0, len(order), per_batch)]
        it = batches
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(batches, desc=f"generate n={n}", leave=False)
            except ImportError:
                pass
        for bi, idx in enumerate(it):
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(dev)
            kw = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id,
                      eos_token_id=self.eos_ids, use_cache=True)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=self.top_k,
                          num_return_sequences=n)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            if self.use_stop_strings:
                kw.update(stop_strings=[s for s in STOP_STRINGS if not s.startswith("<|")], tokenizer=self.tok)
            torch.manual_seed(seed * 100_003 + bi)
            try:
                seqs = self.model.generate(**enc, **kw)
            except (TypeError, ValueError) as e:
                if "stop_strings" not in kw:
                    raise
                warnings.warn(f"stop_strings unsupported ({e}); falling back to post-hoc truncation")
                self.use_stop_strings = False
                kw.pop("stop_strings"), kw.pop("tokenizer")
                seqs = self.model.generate(**enc, **kw)
            new = seqs[:, enc["input_ids"].shape[1]:]
            # count real generated tokens (up to and including first EOS)
            is_eos = torch.zeros_like(new, dtype=torch.bool)
            for e in self.eos_ids:
                is_eos |= new == e
            has_eos = is_eos.any(dim=1)
            first = torch.where(has_eos, is_eos.float().argmax(dim=1) + 1, torch.full_like(has_eos, new.shape[1], dtype=torch.long))
            stats.new_tokens += int(first.sum())
            texts = self.tok.batch_decode(new, skip_special_tokens=True)
            k = n if temperature > 0 else 1
            for j, i in enumerate(idx):
                out[i] = [truncate_at_stop(texts[j * k + s]) for s in range(k)]
            stats.sequences += len(texts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            stats.peak_mem_gb = torch.cuda.max_memory_allocated() / 2**30
        stats.seconds = time.time() - t0
        return out, stats
