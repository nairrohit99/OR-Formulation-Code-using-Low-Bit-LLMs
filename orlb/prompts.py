"""Prompt templates and per-library settings (PuLP or Pyomo) shared by training, sampling, evaluation.

Every generated program must follow one output convention so that grading is exact and does
not depend on guessing which printed number is the answer:

    OBJECTIVE_VALUE: <float>          when the solver reports an optimum
    STATUS: <termination condition>   otherwise (infeasible, unbounded, ...)
"""
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_LIBRARY = "pulp"


@dataclass(frozen=True)
class Library:
    name: str
    solve_snippet: str          # canonical solve + report block shown to the translator
    rules: str                  # library-specific translation rules
    required_markers: tuple     # substrings a legitimate program must contain
    version_check: str          # python code printing an error message if the library is unusable


PULP = Library(
    name="pulp",
    solve_snippet="""model.solve(pulp.PULP_CBC_CMD(msg=False))
if pulp.LpStatus[model.status] == "Optimal":
    print(f"OBJECTIVE_VALUE: {pulp.value(model.objective)}")
    # then print the decision variable values
else:
    print(f"STATUS: {pulp.LpStatus[model.status]}")""",
    rules="""1. Use `import pulp` and build `model = pulp.LpProblem("name", pulp.LpMinimize)` (or `pulp.LpMaximize`).
   Use the classic PuLP 2.x API: `pulp.LpVariable(name, lowBound=..., upBound=..., cat="Continuous" | "Integer" | "Binary")`,
   `pulp.LpVariable.dicts`, `pulp.lpSum`, `model += expression` for the objective and `model += constraint, "name"` for constraints.
2. Keep exactly the same decision variables (bounds and integrality), objective (including min/max sense) and constraints. Do not change any numbers. Use meaningful names.
3. PuLP only supports linear models. If the program has a quadratic or other nonlinear objective or constraint, do not translate it: reply with the single word NOT_LINEAR.""",
    required_markers=("LpProblem", ".solve("),
    version_check=(
        "import pulp\n"
        "v = tuple(int(x) for x in pulp.__version__.split('.')[:2])\n"
        "if v >= (3, 0): print(f'pulp {pulp.__version__} found; this project needs the classic API: pip install pulp==2.9.0')\n"
        "elif not pulp.PULP_CBC_CMD(msg=False).available(): print('pulp is installed but its bundled CBC solver is not available')\n"
    ),
)

PYOMO = Library(
    name="pyomo",
    solve_snippet="""solver = pyo.SolverFactory('appsi_highs')
results = solver.solve(model, load_solutions=False)
tc = results.solver.termination_condition
if tc == pyo.TerminationCondition.optimal:
    model.solutions.load_from(results)
    print(f"OBJECTIVE_VALUE: {pyo.value(model.obj)}")
    # then print the decision variable values
else:
    print(f"STATUS: {tc}")""",
    rules="""1. Use `import pyomo.environ as pyo` and build a `model = pyo.ConcreteModel()`. Name the objective `model.obj`.
2. Keep exactly the same decision variables (bounds and integrality), objective (including min/max sense) and constraints. Do not change any numbers. Use meaningful names.
3. The solver (HiGHS) supports linear, mixed-integer linear and convex continuous quadratic models only. If the program is outside these classes, reply with the single word NOT_SUPPORTED.""",
    required_markers=("SolverFactory", ".solve("),
    version_check=(
        "import pyomo.environ as pyo\n"
        "if not pyo.SolverFactory('appsi_highs').available(exception_flag=False): print('appsi_highs not available: pip install highspy')\n"
    ),
)

LIBRARIES = {"pulp": PULP, "pyomo": PYOMO}


def get_library(name: str) -> Library:
    if name not in LIBRARIES:
        raise ValueError(f"unknown library {name!r}; choose from {sorted(LIBRARIES)}")
    return LIBRARIES[name]


def check_library(name: str, python: str = None) -> None:
    """Fail early (instead of grading every program as a crash) if the library is unusable."""
    lib = get_library(name)
    p = subprocess.run([python or sys.executable, "-c", lib.version_check], capture_output=True, text=True, timeout=120)
    msg = (p.stdout.strip().splitlines() or [""])[-1] if p.returncode == 0 else p.stderr.strip().splitlines()[-1]
    if msg:
        raise RuntimeError(f"[{name}] {msg}")


def code_header(library: str) -> str:
    return f"## Python Code Solution Using `{get_library(library).name}`:"


TASK_TEMPLATE = (
    "Below is an operations research question. Build a mathematical model and "
    "corresponding python code using `{library}` that appropriately addresses the question.\n\n"
    "# Question:\n{question}\n\n# Response:\n"
)

# Strings that mean the model has finished its answer and started a new document.
STOP_STRINGS = [
    "\n# Question:",
    "Below is an operations research question",
    "\n# Execution Feedback:",
    "<|endoftext|>",
    "<|im_end|>",
]

REPAIR_TEMPLATE = (
    "{prompt}{response}\n\n"
    "# Execution Feedback:\n"
    "Running the code above did not produce a valid result.\n"
    "{feedback}\n\n"
    "Fix the model and/or the code. Write the complete corrected response.\n\n"
    "# Corrected Response:\n"
)

TRANSLATE_TEMPLATE = """You are an expert in operations research and the {library} modeling library.
Translate the following COPT (`coptpy`) Python program into an equivalent {library} program.

Requirements:
{rules}
4. Solve and report the result with exactly this pattern:
```python
{solve_snippet}
```
5. Do not hard-code or pre-compute the answer; it must come from the solver.
6. Return only one ```python code block and nothing else.

Problem description (for context):
{question}

coptpy program:
```python
{code}
```"""


def task_prompt(question: str, library: str = DEFAULT_LIBRARY) -> str:
    return TASK_TEMPLATE.format(question=question.strip(), library=get_library(library).name)


def translate_prompt(question: str, code: str, library: str = DEFAULT_LIBRARY) -> str:
    lib = get_library(library)
    return TRANSLATE_TEMPLATE.format(library=lib.name, rules=lib.rules, solve_snippet=lib.solve_snippet,
                                     question=question.strip(), code=code.strip())


def repair_prompt(prompt: str, response: str, feedback: str) -> str:
    return REPAIR_TEMPLATE.format(prompt=prompt, response=response.rstrip(), feedback=feedback.strip())
