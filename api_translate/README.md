# api_translate: coptpy → Pyomo (or PuLP) with Claude / Gemini

This folder turns the **coptpy** solutions in OR-Instruct-Data-3K into **Pyomo** solutions using a
cloud model (Claude or Gemini). Its output is the training data for your quantized Pyomo models.

Every translation is **checked by running it**. A translation is kept only if it gives the same
optimal objective value as the original coptpy program. When the value is wrong, or the code
crashes, the error is sent back to the model and it tries again (up to `--rounds` times). The
reference value itself is never shown to the model, so it cannot just print the number.

## Where to run it
On any machine **with internet**. It needs a CPU only, no GPU. It must have:

```bash
pip install -r api_translate/requirements.txt
```

You also need `data/or_instruct_ref.jsonl`. This is the reference answer for each problem, made by
running the original coptpy code (`coptpy` from pip works without a license for problems this size):

```bash
python scripts/s00_reference_answers.py --or_instruct <path to OR-Instruct-Data-3K>
```

When it finishes, copy `data/pyomo/` to the offline A100 machine.

## API keys
Keys are read from environment variables. Don't put keys in the code or in the repo.

| Provider | Variable |
|---|---|
| Claude | `ANTHROPIC_API_KEY` (or log in once with `ant auth login`) |
| Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |

## Usage

```bash
# 1. See the exact prompt and the number of requests. Nothing is sent, nothing is paid.
python api_translate/translate.py --provider claude --dry_run --limit 3

# 2. Pilot on 50 problems, then check the yield and the printed token totals.
python api_translate/translate.py --provider claude --library pyomo --limit 50

# 3. Full run. Claude: the default model is claude-opus-5-5.
python api_translate/translate.py --provider claude --library pyomo

#    Same, via Message Batches: half price, each round usually finishes within an hour.
python api_translate/translate.py --provider claude --library pyomo --batch

#    Gemini: pass the exact model id from Google AI Studio.
python api_translate/translate.py --provider gemini --model <gemini-model-id> --library pyomo
```

Output (the same format as `scripts/s01_translate.py`, so the rest of the pipeline works unchanged):

```
data/pyomo/or_instruct.jsonl   every problem: reference answer, verified Pyomo answer (or null), attempts
data/pyomo/sft_train.jsonl     verified (prompt, completion) pairs, ready for training
data/pyomo/sft_val.jsonl
```

Then, on the A100 machine:

```bash
python scripts/s02_sft.py --base ../models/Qwen3-4B-Base --quant nf4 \
    --train data/pyomo/sft_train.jsonl --val data/pyomo/sft_val.jsonl --out runs/sft_pyomo_nf4
```

## Useful options
| Option | Meaning |
|---|---|
| `--library pyomo\|pulp` | Target library (default `pyomo` in this folder) |
| `--rounds 3` | Attempts per problem. Rounds 2+ include the solver's feedback |
| `--effort medium` | Claude thinking effort. Try `high` if many problems stay unverified (costs more) |
| `--model` | Claude: any model id (default `claude-opus-5-5`). Gemini: required |
| `--workers 8` | Parallel API calls. Lower it if you hit rate limits |
| `--no_fallbacks` | Claude: turn off the server-side refusal fallback (on by default; see below) |
| `--limit N` | Only the first N problems |

## Cost and safety notes
- **The cache saves you money on reruns.** Every response is saved in `api_translate/cache/`. Rerunning
  the same command (after a crash, Ctrl-C, or with more `--rounds`) only pays for requests not made
  before. Transient API errors are not cached, so a rerun retries them.
- **Estimate the cost from the pilot.** Run `--limit 50` first. The script prints input and output token
  totals; multiply by about 3000/50 and by your model's price per token. Later rounds only re-send the
  problems that failed.
- **Refusal fallback (Claude).** If a request is declined by a safety filter, the API reruns it on a
  fallback model chosen by Anthropic (`fallbacks: "default"`), so the problem is not lost. Batch mode
  cannot use this. Turn it off with `--no_fallbacks`.
- **Your data leaves your machine.** The OR-Instruct questions and code are sent to the API provider.
  They are public data, but keep that in mind if you later add your own problems.
- **Model scope.** Problems that need a nonlinear solver are answered with `NOT_SUPPORTED` and skipped
  (counted as `refused`). HiGHS handles LP, MILP and convex continuous QP.

## For the paper
State that a cloud LLM was used **only to change the solver language** of existing, human-checked
answers, that every kept translation reproduced the coptpy objective value, and how many were kept
(`verified_by` in the printed stats). Name the provider and model; they are also saved in the
`translator` field of every row.
