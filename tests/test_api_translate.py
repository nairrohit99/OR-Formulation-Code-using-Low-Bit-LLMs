"""api_translate/translate.py with a fake provider (no network, no API key):
feedback retries, refusals, output files, and that a rerun is served entirely from the cache."""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "api_translate"))
from providers import Completion  # noqa: E402

from test_core import DUCKS_PYOMO  # noqa: E402

spec = importlib.util.spec_from_file_location("api_translate_main", os.path.join(ROOT, "api_translate", "translate.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

GOOD = "```python\n" + DUCKS_PYOMO + "```"
WRONG = GOOD.replace("20*model.boat", "25*model.boat")


class FakeProvider:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if "Quadratic" in prompt:
            return Completion(text="NOT_SUPPORTED", input_tokens=10, output_tokens=1)
        if "previous translation attempt" in prompt:
            assert "does NOT match" in prompt and "1160" not in prompt.split("Problem with that attempt:")[1]
            return Completion(text=GOOD, input_tokens=10, output_tokens=50)
        return Completion(text=WRONG, input_tokens=10, output_tokens=50)


def _ref_rows():
    comp = "## Mathematical Model:\n- x\n\n## Python Code Solution Using `coptpy`:\n```python\nimport coptpy\n```"
    rows = []
    for i, q in enumerate(["Ducks problem A.", "Ducks problem B.", "Quadratic portfolio problem."]):
        rows.append({"id": f"ori-{i}", "question": q, "orig_prompt": "", "orig_completion": comp,
                     "orig_code": "import coptpy  # placeholder", "ref_status": "ok", "ref_objective": 1160.0})
    rows.append({"id": "ori-3", "question": "Broken ref.", "orig_prompt": "", "orig_completion": comp,
                 "orig_code": "x", "ref_status": "runtime_error", "ref_objective": None})
    return rows


def _run(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(tr, "make_provider", lambda *a, **k: fake)
    ref = tmp_path / "ref.jsonl"
    ref.write_text("".join(json.dumps(r) + "\n" for r in _ref_rows()))
    monkeypatch.setattr(sys, "argv", ["translate.py", "--provider", "claude", "--library", "pyomo", "--ref", str(ref),
                                      "--out_dir", str(tmp_path / "out"), "--cache_dir", str(tmp_path / "cache"),
                                      "--rounds", "3", "--val_frac", "0", "--workers", "2"])
    tr.main()
    return [json.loads(l) for l in open(tmp_path / "out" / "or_instruct.jsonl")]


def test_translate_feedback_cache(tmp_path, monkeypatch):
    fake = FakeProvider()
    items = _run(tmp_path, fake, monkeypatch)
    by_id = {it["id"]: it for it in items}
    for i in ("ori-0", "ori-1"):  # wrong first, fixed after feedback
        assert by_id[i]["verified_by"] == "reference" and len(by_id[i]["attempt_log"]) == 2
        assert "Using `pyomo`" in by_id[i]["completion"] and "using `pyomo`" in by_id[i]["prompt"]
    assert by_id["ori-2"]["refused"] and by_id["ori-2"]["completion"] is None
    assert by_id["ori-3"]["attempt_log"] == []  # no reference answer -> not sent
    assert fake.calls == 5  # 3 first-round + 2 feedback
    sft = [json.loads(l) for l in open(tmp_path / "out" / "sft_train.jsonl")]
    assert len(sft) == 2

    fake2 = FakeProvider()  # same command again: everything comes from the cache
    items2 = _run(tmp_path, fake2, monkeypatch)
    assert fake2.calls == 0
    assert sum(it["verified_by"] == "reference" for it in items2) == 2
