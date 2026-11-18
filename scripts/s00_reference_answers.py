"""Step 0: run the original coptpy reference code of OR-Instruct-Data-3K to get a ground-truth
objective value for every training problem (the dataset has no answer column).

Needs `coptpy` (with its license) in the python given by --python. CPU only.

python scripts/s00_reference_answers.py --or_instruct ../datasets/OR-Instruct-Data-3K --out data/or_instruct_ref.jsonl
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orlb.data import load_or_instruct, save_jsonl  # noqa: E402
from orlb.execution import run_many  # noqa: E402

# Records every coptpy model so we can print the solver's own objective value at exit,
# instead of guessing which printed number is the objective.
PRELUDE = r'''
import atexit as _orlb_atexit
_ORLB_MODELS = []
try:
    import coptpy as _orlb_cp
    _orlb_orig_create = _orlb_cp.Envr.createModel
    def _orlb_create(self, *a, **k):
        _m = _orlb_orig_create(self, *a, **k)
        _ORLB_MODELS.append(_m)
        return _m
    _orlb_cp.Envr.createModel = _orlb_create
except Exception:
    pass

def _orlb_report():
    try:
        import coptpy as _cp
        C = _cp.COPT
        for _m in reversed(_ORLB_MODELS):
            st = _m.status
            if st == C.OPTIMAL:
                print(f"OBJECTIVE_VALUE: {_m.objval}"); return
            if st == C.INFEASIBLE:
                print("STATUS: infeasible"); return
            if st == C.UNBOUNDED:
                print("STATUS: unbounded"); return
            if st == getattr(C, "INF_OR_UNB", -999):
                print("STATUS: infeasible_or_unbounded"); return
    except Exception:
        pass
_orlb_atexit.register(_orlb_report)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--or_instruct", required=True, help="HF dataset dir / jsonl / parquet of OR-Instruct-Data-3K")
    ap.add_argument("--out", default="data/or_instruct_ref.jsonl")
    ap.add_argument("--python", default=sys.executable, help="python interpreter that has coptpy")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = load_or_instruct(args.or_instruct)[: args.limit]
    print(f"{len(items)} problems; running reference code ...")
    res = run_many([it["orig_code"] for it in items], timeout=args.timeout, workers=args.workers,
                   python=args.python, prelude=PRELUDE)
    for it, r in zip(items, res):
        it.update(ref_status=r.status, ref_objective=r.objective, ref_parse_mode=r.parse_mode,
                  ref_error=r.error_type, ref_stdout_tail=r.stdout[-500:], ref_stderr_tail=r.stderr[-500:])
    save_jsonl(items, args.out)
    print("status:", Counter(r.status for r in res))
    print("parse mode:", Counter(r.parse_mode for r in res))
    if any(r.error_type == "ModuleNotFoundError" for r in res):
        print("WARNING: ModuleNotFoundError seen -- is coptpy installed for", args.python, "?")
    print("saved", args.out)


if __name__ == "__main__":
    main()
