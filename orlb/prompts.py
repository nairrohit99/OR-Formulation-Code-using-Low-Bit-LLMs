"""Prompt templates shared by training, sampling and evaluation.

Every generated program must follow one output convention so that grading is
exact and does not depend on guessing which printed number is the answer:

    OBJECTIVE_VALUE: <float>          when the solver reports an optimum
    STATUS: <termination condition>   otherwise (infeasible, unbounded, ...)
"""

SOLVER_NAME = "appsi_highs"

TASK_TEMPLATE = (
    "Below is an operations research question. Build a mathematical model and "
    "corresponding python code using `pyomo` that appropriately addresses the question.\n\n"
    "# Question:\n{question}\n\n# Response:\n"
)

CODE_HEADER = "## Python Code Solution Using `pyomo`:"

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

PYOMO_SOLVE_SNIPPET = f"""solver = pyo.SolverFactory('{SOLVER_NAME}')
results = solver.solve(model, load_solutions=False)
tc = results.solver.termination_condition
if tc == pyo.TerminationCondition.optimal:
    model.solutions.load_from(results)
    print(f"OBJECTIVE_VALUE: {{pyo.value(model.obj)}}")
    # then print the decision variable values
else:
    print(f"STATUS: {{tc}}")"""

TRANSLATE_TEMPLATE = """You are an expert in operations research and the Pyomo modeling language.
Translate the following COPT (`coptpy`) Python program into an equivalent Pyomo program.

Requirements:
1. Use `import pyomo.environ as pyo` and build a `model = pyo.ConcreteModel()`.
2. Keep exactly the same decision variables (bounds and integrality), objective (including min/max sense) and constraints. Do not change any numbers. Use meaningful names.
3. Name the objective `model.obj`.
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


def task_prompt(question: str) -> str:
    return TASK_TEMPLATE.format(question=question.strip())


def translate_prompt(question: str, code: str) -> str:
    return TRANSLATE_TEMPLATE.format(
        solve_snippet=PYOMO_SOLVE_SNIPPET, question=question.strip(), code=code.strip()
    )


def repair_prompt(prompt: str, response: str, feedback: str) -> str:
    return REPAIR_TEMPLATE.format(prompt=prompt, response=response.rstrip(), feedback=feedback.strip())
