"""Pipeline logic with a scripted fake generator (no model needed): grading, voting, repair, scoring."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orlb.modeling import GenStats  # noqa: E402
from orlb.pipeline import score_records, solve_items  # noqa: E402

from test_core import DUCKS  # noqa: E402

GOOD = "## Mathematical Model:\n- x\n\n```python\n" + DUCKS + "```"
WRONG = GOOD.replace("20*model.boat", "25*model.boat")       # runs, wrong objective value
CRASH = GOOD.replace("model.obj = ", "model.obj = undefined_name + ")


class FakeGen:
    """Initial calls return the scripted samples; repair prompts get GOOD back."""

    def __init__(self, script):
        self.script = script

    def generate(self, prompts, n=1, temperature=0.0, top_p=1.0, seed=0, progress=True):
        out = []
        for p in prompts:
            if "# Corrected Response:" in p:
                out.append([GOOD])
            else:
                out.append(self.script[p][:n])
        return out, GenStats(sequences=len(prompts) * n, new_tokens=10, seconds=0.1)


def test_vote_and_repair():
    items = [{"id": "a", "answer": 1160}, {"id": "b", "answer": "1160"}, {"id": "c", "answer": 1160}]
    prompts = ["pa", "pb", "pc"]
    script = {
        "pa": [GOOD, GOOD, WRONG, CRASH],    # first correct; maj correct
        "pb": [WRONG, WRONG, GOOD, CRASH],   # first wrong; maj wrong; pass@4 correct
        "pc": [CRASH, GOOD, GOOD, WRONG],    # first crashes -> fixed by repair
    }
    recs, cost = solve_items(FakeGen(script), items, prompts, n=4, temperature=0.7, repair_rounds=1)
    before = score_records(recs, ks=(1, 4), which="first")
    after = score_records(recs, ks=(1, 4), which="final")
    assert abs(before["first_sample_acc"] - 1 / 3) < 1e-9
    assert abs(before["maj@4"] - 2 / 3) < 1e-9  # a: GOOD x2; c: GOOD x2 vs WRONG x1
    assert before["pass@4"] == 1.0
    assert abs(after["first_sample_acc"] - 2 / 3) < 1e-9  # c repaired
    assert before["breakdown_first_sample"] == {"correct": 1, "wrong_value": 1, "runtime_error": 1}
    assert recs[2]["samples"][0]["attempts"][1]["kind"] == "repair1"
    assert cost["repair"]["sequences"] == 3  # three CRASH samples were repaired
