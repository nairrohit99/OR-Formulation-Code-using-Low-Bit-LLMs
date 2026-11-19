# Cheap Model, Free Checker: low-bit LLMs for OR modeling with solver feedback

**Research question.** Can a **4-bit Qwen3-4B** model, improved with **solver-verified self-training**
(rejection sampling + DPO) and run with **solver-checked voting and repair**, match larger
full-precision models at writing optimization models from text, using much less memory?

**Main idea.** For an OR problem, the solver is a perfect, free grader: if the generated program runs
and returns the right objective value, the answer is right. A 4-bit model is cheap enough to answer many
times. The solver then picks the good answers, both for training and at test time.

The original course project (QLoRA SFT on OR-Instruct-3K, coptpy) is in `legacy/` and is kept for reference only.

## Solver library: `--library pulp` (default) or `--library pyomo`

| | PuLP 2.9 + CBC (default) | Pyomo + HiGHS |
|---|---|---|
| LP / MILP | yes | yes |
| Continuous quadratic problems | no (dropped) | yes |
| Nonlinear problems | no | no |
| Solver install | bundled in the pip wheel | `highspy` |

**Pin `pulp==2.9.0`.** PuLP 4.x is a rewrite (new Rust core) that removes the classic API
(`LpVariable(..., lowBound=, cat=)`, `PULP_CBC_CMD`). The scripts check this at startup and stop
with a clear message.

Choose the library in step 1 (`s01_translate.py --library ...`). Its output goes to `data/<library>/`,
and every later step follows from those files. `s06_evaluate.py` takes `--library` and it must match the
adapter. Training on files from two libraries at once is refused.

## Pipeline

| Step | Script | What it does | Hardware |
|---|---|---|---|
| 0 | `s00_reference_answers.py` | Runs the original **coptpy** code of OR-Instruct-3K to get each problem's true objective value (the dataset has no answer column) | CPU, needs `coptpy` |
| 1 | `s01_translate.py` | A larger local LLM translates each coptpy program to **PuLP** (or Pyomo). A translation is kept **only if it reproduces the coptpy objective value**. Problems the translator marks as nonlinear are flagged `refused` and skipped | GPU |
| 2 | `s02_sft.py` | QLoRA (nf4) or LoRA (bf16) SFT with fixed masking, plus a validation split | GPU |
| 3 | `s03_sample.py` | Samples k answers per training problem, runs them, grades them against the reference, and optionally does one repair round | GPU + CPU |
| 4 | `s04_build_training_sets.py` | Builds the **RFT** set (self-generated answers the solver confirmed correct) and **DPO pairs** (correct vs. wrong for the same problem) | CPU |
| 4b | `s02_sft.py --train data/pulp/sft_train.jsonl data/pulp/rft.jsonl` | Rejection-sampling fine-tuning (RFT) | GPU |
| 5 | `s05_dpo.py` | DPO with an NLL term on the chosen answer, starting from the RFT adapter | GPU |
| 6 | `s06_evaluate.py` | Benchmarks: greedy, k-sample majority vote, repair rounds; logs memory, tokens and time | GPU + CPU |
| 7 | `s07_error_analysis.py` | Error taxonomy, plus a CSV for labelling errors by hand | CPU |
| 8 | `s08_make_tables.py` | Tables with 95% CIs and paired McNemar tests | CPU |
| – | `check_overlap.py` | Checks for overlap (contamination) between training and test questions | CPU |

### Output convention (used for exact grading)
Every generated program prints `OBJECTIVE_VALUE: <number>`, or `STATUS: <Infeasible|Unbounded|...>`, whichever library it uses.
The SFT data teaches this convention. For programs that ignore it, the parser falls back to keyword
matching, and `parse_modes` in the summary reports how often that fallback was used.

## Bugs fixed relative to `legacy/`
1. **The model never learned to stop.** `DataCollatorForLanguageModeling` masks every token equal to
   `pad_token`, and for Qwen base `pad == eos`, so the final EOS was never trained. Generation then ran
   on to 8192 tokens. The new collator masks padding **by position**.
2. **Inference appended EOS to the prompt** (`prompt + eos`), which never happens in training.
3. **All python blocks were concatenated**, so run-on generations executed code for invented extra
   questions. Now only the first block runs.
4. **The loss was computed on the prompt too.** It is now computed on the completion only.
5. **Checkpoints were selected by training loss.** They are now selected by held-out validation loss.
6. Generated code now runs with a **timeout** in an isolated temp directory, and the model is loaded once, not once per question.

## Experiments for the paper

| ID | Model | Training | Inference | Purpose |
|---|---|---|---|---|
| E1 | 4B nf4 | none (3-shot) | greedy | lower bound |
| E2 | 4B **bf16** | SFT | greedy | full-precision reference (claimed in the report, never measured) |
| E3 | 4B nf4 | SFT | greedy | fixed version of the course project |
| E4 | 4B nf4 | SFT → RFT | greedy | value of solver-verified self-training |
| E5 | 4B nf4 | SFT → RFT → DPO | greedy | value of preference learning |
| E6 | E3–E5 | – | maj@{2,4,8}, repair 0/1/2 | test-time compute with the solver as checker |
| E7 | 8B **bf16** SFT vs. 4B nf4 E5 + k=8 | – | – | **budget-matched comparison** (peak GB, s/problem, tokens) |
| E8 | E2 vs. E3 | – | – | **which errors does 4-bit introduce?** (`s07`) |
| E9 | 4B bf16 | SFT → RFT → DPO | greedy | does self-training close the quantization gap? (optional) |
| E10 | E3 with `--library pyomo` | SFT | greedy | **does the solver library matter for small 4-bit models?** Compare crash rate and accuracy on the LP/MILP problems both libraries cover |

Ablations: DPO `--sft_weight 0` vs. `1`; `--gold_chosen` on/off; `--include_repairs` on/off; the old
problem-category labels, reported as a negative result.

Benchmarks: NL4Opt (`data/nl4opt_test.jsonl`, 230 problems), MAMO EasyLP / ComplexLP, IndustryOR.
Always report **all** problems, never a subset. Report strict accuracy (relative tolerance 1e-4) and the
rounding-lenient variant side by side.

## Commands (offline A100 80GB)

```bash
pip install -r requirements.txt            # offline: pip install --no-index --find-links ../wheels -r requirements.txt
python -m pytest -q tests                  # CPU sanity checks (pulp 2.9, pyomo, highspy)
B=../models/Qwen3-4B-Base                  # full-precision weights; nf4 is applied on load

# 0-1: data
python scripts/s00_reference_answers.py --or_instruct ../datasets/OR-Instruct-Data-3K
python scripts/s01_translate.py --library pulp --translator ../models/Qwen3-32B --quant nf4 --rounds 4
python scripts/check_overlap.py --train data/pulp/or_instruct.jsonl --bench nl4opt=data/nl4opt_test.jsonl

# 2: SFT (4-bit and bf16)
python scripts/s02_sft.py --base $B --quant nf4  --out runs/sft_nf4
python scripts/s02_sft.py --base $B --quant none --out runs/sft_bf16

# 3-4: self-training data, then RFT
python scripts/s03_sample.py --base $B --quant nf4 --adapter runs/sft_nf4/final --data data/pulp/or_instruct.jsonl \
       --n 8 --out data/pulp/samples_sft_nf4.jsonl
python scripts/s04_build_training_sets.py --samples data/pulp/samples_sft_nf4.jsonl --gold_chosen \
       --rft_out data/pulp/rft_nf4.jsonl --dpo_out data/pulp/dpo_nf4.jsonl
python scripts/s02_sft.py --base $B --quant nf4 --train data/pulp/sft_train.jsonl data/pulp/rft_nf4.jsonl --out runs/rft_nf4

# 5: DPO (optionally re-sample with the RFT model first and rebuild the pairs)
python scripts/s05_dpo.py --base $B --quant nf4 --adapter runs/rft_nf4/final --pairs data/pulp/dpo_nf4.jsonl --out runs/dpo_nf4

# 6: evaluation
BENCH="nl4opt=data/nl4opt_test.jsonl mamo_easy=../datasets/MAMO/easy.jsonl mamo_complex=../datasets/MAMO/complex.jsonl industryor=../datasets/IndustryOR/test.jsonl"
python scripts/s06_evaluate.py --base $B --quant nf4 --library pulp --adapter runs/dpo_nf4/final --bench $BENCH --tag dpo_nf4_greedy
python scripts/s06_evaluate.py --base $B --quant nf4 --library pulp --adapter runs/dpo_nf4/final --bench $BENCH --n 8 --temperature 0.7 --repair_rounds 1 --tag dpo_nf4_k8_rep1
python scripts/s06_evaluate.py --base $B --quant nf4 --library pulp --bench $BENCH --fewshot_file data/pulp/sft_train.jsonl --fewshot_k 3 --tag base_nf4_3shot

# 7-8: analysis and tables
python scripts/s07_error_analysis.py --runs results/sft_bf16_greedy results/sft_nf4_greedy --bench nl4opt
python scripts/s08_make_tables.py --runs "results/*" --baseline results/sft_nf4_greedy
```

The MAMO and IndustryOR paths above are placeholders. Point them at whatever files you have; the loader
detects `question`/`en_question` and `answer`/`en_answer` automatically.

Rough time on one A100 80GB with HF `generate`:
- Translation: a few hours.
- Each SFT/RFT run: about 1–2 h.
- Sampling 3K × 8: about 4–6 h (`--batch_size 64`; resumable).
- DPO: about 1–2 h.
- Each greedy evaluation over all benchmarks: under 1 h.

## Things to be honest about in the paper
- Solver feedback for OR modeling is not new (e.g. ORLM's data pipeline, LLMOPT's self-correction,
  solver-informed RL). The contribution here is the **low-bit, budget-matched** angle plus the
  **quantization error analysis**. Cite ORLM, the source of OR-Instruct and the NL4Opt split used here.
- With PuLP, quadratic and nonlinear problems are out of scope. Report how many training problems were
  `refused`, and how many benchmark problems are nonlinear (they count as wrong for every system).
- Step 1 uses a larger LLM only to **convert the solver language** of existing answers. Every kept
  translation is verified against the original solver's objective. State this, and report the yield.
- The public test sets contain some wrong labels. Report the number of `ground_truth_suspect` cases found during manual labelling.
