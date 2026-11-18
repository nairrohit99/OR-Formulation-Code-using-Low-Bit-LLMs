"""Dataset loading / saving helpers."""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Iterable, Optional

from .execution import extract_code
from .prompts import CODE_HEADER

Q_KEYS = ["question", "en_question", "Question", "problem", "description", "prompt"]
A_KEYS = ["answer", "en_answer", "Answer", "optimal_value", "ground_truth", "objective"]


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(rows: Iterable[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(row: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_key(row: dict, keys: list[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return k
    return None


def load_benchmark(path: str, name: Optional[str] = None) -> list[dict]:
    """Load a test set (.jsonl/.json/.parquet/HF dir) into [{id, question, answer}]."""
    rows = _load_any(path)
    name = name or os.path.splitext(os.path.basename(path.rstrip("/")))[0]
    out = []
    for i, r in enumerate(rows):
        qk, ak = _first_key(r, Q_KEYS), _first_key(r, A_KEYS)
        if qk is None:
            raise KeyError(f"{path}: no question field in row {i}; keys={list(r)}")
        q = r[qk]
        if qk == "prompt":
            q = extract_question(q)
        out.append({"id": f"{name}-{i}", "question": str(q).strip(), "answer": r.get(ak) if ak else None})
    return out


def _load_any(path: str) -> list[dict]:
    if os.path.isdir(path) or path.endswith(".parquet"):
        from datasets import load_dataset, load_from_disk
        try:
            ds = load_from_disk(path)
            if hasattr(ds, "keys"):
                ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        except Exception:
            ds = load_dataset(path, split="train") if os.path.isdir(path) else load_dataset("parquet", data_files=path, split="train")
        return [dict(r) for r in ds]
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else list(data.values())
    return load_jsonl(path)


def load_or_instruct(path: str) -> list[dict]:
    """OR-Instruct-Data-3K -> [{id, question, orig_prompt, orig_completion, orig_code}]. Keeps every row."""
    rows = _load_any(path)
    out = []
    for i, r in enumerate(rows):
        prompt, completion = r.get("prompt", ""), r.get("completion", "")
        out.append({
            "id": f"ori-{i}",
            "question": extract_question(prompt),
            "orig_prompt": prompt,
            "orig_completion": completion,
            "orig_code": extract_code(completion),
        })
    return out


_Q_RE = re.compile(r"#\s*Question:\s*(.*?)\s*#\s*Response:", re.S)


def extract_question(prompt: str) -> str:
    m = _Q_RE.search(prompt or "")
    return (m.group(1) if m else (prompt or "")).strip()


_PY_HEADER = re.compile(r"\n?#+\s*Python Code.*", re.S | re.I)


def math_part(completion: str) -> str:
    """Everything before the code section (the mathematical model write-up)."""
    text = completion or ""
    m = _PY_HEADER.search(text)
    if m:
        text = text[: m.start()]
    elif "```" in text:
        text = text[: text.index("```")]
    text = re.sub(r"`?coptpy`?|COPT", "pyomo", text)
    return text.rstrip()


def build_completion(math_text: str, code: str) -> str:
    return f"{math_text.rstrip()}\n\n{CODE_HEADER}\n```python\n{code.strip()}\n```"


def split_of(key: str, val_frac: float, seed: int = 0) -> str:
    h = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest(), 16) % 10_000
    return "val" if h < val_frac * 10_000 else "train"
