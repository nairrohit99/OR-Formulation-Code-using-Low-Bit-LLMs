"""CPU unit tests: code extraction, execution, parsing, grading, voting.
Needs pulp==2.9.0, pyomo and highspy.  Run: python -m pytest -q tests
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orlb.data import build_completion, extract_question, math_part, split_of  # noqa: E402
from orlb.execution import extract_code, parse_output, run_code  # noqa: E402
from orlb.grading import NO_SOL, majority_vote, match, mcnemar_exact, parse_gt, wilson_ci  # noqa: E402
from orlb.prompts import LIBRARIES, PULP, PYOMO, check_library, task_prompt, translate_prompt  # noqa: E402

DUCKS_PULP = """import pulp
model = pulp.LpProblem("ducks", pulp.LpMinimize)
boat = pulp.LpVariable("boat", lowBound=0, cat="Integer")
canoe = pulp.LpVariable("canoe", lowBound=0, cat="Integer")
model += 20*boat + 40*canoe
model += 10*boat + 8*canoe >= 300, "ducks"
model += boat <= 12, "max_boat"
model += canoe >= 0.6*(boat + canoe), "canoe_share"
""" + PULP.solve_snippet + "\n"

INFEASIBLE_PULP = """import pulp
model = pulp.LpProblem("inf", pulp.LpMinimize)
x = pulp.LpVariable("x", lowBound=0)
model += x
model += x <= -1
""" + PULP.solve_snippet + "\n"

DUCKS_PYOMO = """import pyomo.environ as pyo
model = pyo.ConcreteModel()
model.boat = pyo.Var(domain=pyo.NonNegativeIntegers)
model.canoe = pyo.Var(domain=pyo.NonNegativeIntegers)
model.obj = pyo.Objective(expr=20*model.boat + 40*model.canoe, sense=pyo.minimize)
model.ducks = pyo.Constraint(expr=10*model.boat + 8*model.canoe >= 300)
model.maxboat = pyo.Constraint(expr=model.boat <= 12)
model.canoe_share = pyo.Constraint(expr=model.canoe >= 0.6*(model.boat + model.canoe))
""" + PYOMO.solve_snippet + "\n"

INFEASIBLE_PYOMO = """import pyomo.environ as pyo
model = pyo.ConcreteModel()
model.x = pyo.Var(domain=pyo.NonNegativeReals)
model.obj = pyo.Objective(expr=model.x)
model.c = pyo.Constraint(expr=model.x <= -1)
""" + PYOMO.solve_snippet + "\n"

DUCKS = DUCKS_PULP  # default library
PROGRAMS = {"pulp": (DUCKS_PULP, INFEASIBLE_PULP), "pyomo": (DUCKS_PYOMO, INFEASIBLE_PYOMO)}


@pytest.mark.parametrize("library", sorted(LIBRARIES))
def test_library_available(library):
    check_library(library)


@pytest.mark.parametrize("library", sorted(LIBRARIES))
def test_snippet_optimal(library):
    r = run_code(PROGRAMS[library][0], timeout=60)
    assert r.status == "ok", (r.status, r.stdout, r.stderr)
    assert abs(r.objective - 1160) < 1e-6 and r.parse_mode == "marker"
    assert all(m in PROGRAMS[library][0] for m in LIBRARIES[library].required_markers)


@pytest.mark.parametrize("library", sorted(LIBRARIES))
def test_snippet_infeasible(library):
    r = run_code(PROGRAMS[library][1], timeout=60)
    assert r.status == "infeasible", (r.status, r.stdout, r.stderr)


def test_pulp_unbounded():
    code = """import pulp
model = pulp.LpProblem("unb", pulp.LpMaximize)
w = pulp.LpVariable("w", lowBound=0)
model += w
model += w >= 1
""" + PULP.solve_snippet
    assert run_code(code).status == "unbounded"


def test_prompts_name_library():
    for lib in LIBRARIES:
        assert f"using `{lib}`" in task_prompt("Q?", lib)
        t = translate_prompt("Q?", "import coptpy", lib)
        assert LIBRARIES[lib].solve_snippet in t and "import coptpy" in t
    assert "NOT_LINEAR" in translate_prompt("Q?", "x", "pulp")
    with pytest.raises(ValueError):
        task_prompt("Q?", "gurobi")


def test_errors():
    assert run_code(None).status == "no_code"
    assert run_code("x = (").status == "syntax_error"
    r = run_code("import nonexistent_mod_xyz")
    assert r.status == "runtime_error" and r.error_type == "ModuleNotFoundError"
    assert run_code("import time\ntime.sleep(10)", timeout=1).status == "timeout"
    assert run_code("print('hello')").status == "no_objective"


def test_extract_code():
    resp = "## Mathematical Model:\n$$x$$\n\n## Python Code\n```python\nprint(1)\n```\ntext\n```python\nprint(2)\n```"
    assert extract_code(resp).strip() == "print(1)"
    assert extract_code("```python\nprint(3)\n").strip() == "print(3)"  # unclosed (truncated)
    assert extract_code("no code here") is None


def test_parse_fallbacks():
    assert parse_output("Minimum total cost: 1,160.00\n", "", 0).objective == 1160.0
    assert parse_output("Model is infeasible\n", "", 0).status == "infeasible"
    assert parse_output("STATUS: infeasibleOrUnbounded\n", "", 0).status == "infeasible_or_unbounded"
    assert parse_output("OBJECTIVE_VALUE: -3.5e2\nx = 4\n", "", 0).objective == -350.0


def test_grading():
    assert parse_gt(1160) == (1160.0, 0)
    assert parse_gt("12.35")[0] == 12.35 and parse_gt("12.35")[1] == 2
    assert parse_gt("No Best Solution")[0] == NO_SOL
    assert parse_gt(NO_SOL)[0] == NO_SOL
    assert match(1160.0000001, 1160.0)
    assert not match(1161, 1160.0)
    assert not match(12.3456, 12.35, 2) and match(12.3456, 12.35, 2, lenient=True)
    assert match(NO_SOL, NO_SOL) and not match(5.0, NO_SOL)


def test_vote_and_stats():
    assert majority_vote([1.0, 2.0, 2.0000001, None, NO_SOL]) == 2.0
    assert majority_vote([None, None]) is None
    assert majority_vote([3.0, 4.0]) == 3.0  # tie -> earliest
    lo, hi = wilson_ci(49, 60)
    assert 0.69 < lo < 0.71 and 0.89 < hi < 0.91
    n01, n10, p = mcnemar_exact([1, 1, 1, 0], [0, 0, 0, 0])
    assert (n01, n10) == (3, 0) and abs(p - 0.25) < 1e-9


def test_data_helpers():
    prompt = ("Below is an operations research question. Build a mathematical model and corresponding python code "
              "using `coptpy` that appropriately addresses the question.\n\n# Question:\nA farm grows wheat.\n\n# Response:\n")
    assert extract_question(prompt) == "A farm grows wheat."
    comp = "## Mathematical Model:\n### Decision Variables\n- x\n\n## Python Code Solution Using `coptpy`:\n```python\nimport coptpy\n```"
    for lib in LIBRARIES:
        mp = math_part(comp, lib)
        assert "coptpy" not in mp and mp.startswith("## Mathematical Model:") and "```" not in mp
        full = build_completion(mp, "print(1)", lib)
        assert extract_code(full).strip() == "print(1)" and f"Using `{lib}`" in full
    assert split_of("a", 0.0) == "train" and split_of("a", 1.0) == "val"
