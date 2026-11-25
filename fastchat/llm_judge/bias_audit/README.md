# Bias audit: GPT-4 vs gpt-6-luna as MT-Bench judge

Plan: (1) justify replacing GPT-4 with gpt-6-luna, (2) audit Luna for the biases Zheng et al. (2023)
found in GPT-4, using the same tests, (3) apply the team's improvements only to biases that show up.

Run everything from `fastchat/llm_judge`.

## 1. No API needed (already done, results committed)

```bash
pip install pandas scipy matplotlib
python -m bias_audit.analyze_existing
```
Writes `figures/01..12_*.png` (16:9, slide-ready) and `results/summary.json`, `results/*.csv`.

| Figure | What it shows |
|---|---|
| 01–06, 08–10 | GPT-4 vs Luna: scores, turns, categories, pairwise, agreement, ranking, Luna errors |
| 07, 11 | Position bias on real pairs (paper Tables 2, 10, 11 style) |
| 12 | Length preference on real answers (style-control regression, γ) |

Data notes:
- `gpt-4_*.jsonl` are the judgments **published by the paper's authors** (timestamps June 2023), not re-run by us.
- `gpt-6-luna_*.jsonl` were run by the team (Nov 2026) with `common.chat_completion_openai`:
  `reasoning_effort="none"`, `max_completion_tokens=2048`, **no temperature** (gpt-6 models do not take one),
  so Luna is not guaranteed deterministic, unlike the paper's temperature 0.
- Prompts sent to both judges are identical (checked on all 800 items).

## ⚠ Prompt confound found (25 Nov 2026)

`data/judge_prompts.jsonl` had been modified before the Luna run: in all four **pairwise** prompts the paper's
sentence "Do not allow the length of the responses to influence your evaluation." was replaced by an
"ANTI-VERBOSITY: … a shorter answer should win when it is equally accurate and complete." instruction.
The **single-answer** prompts were untouched (identical to upstream FastChat).

- GPT-4 pairwise judgments (authors, 2023) used the original prompt. Luna's pairwise judgments most likely used
  the modified one (it says "concise" in 8.9% of pairwise vs 3.6% of single judgments; GPT-4 shows the opposite).
- Fixed on this branch: the original prompts are restored; the modified ones are kept as `*-antiverbosity`.
- Single-answer comparisons are clean. Pairwise Luna results (length γ, position, agreement, turn 2) are provisional.

Clean re-run of Luna pairwise with the paper's prompt (1,600 calls), keeping the old run as a prompt-fix experiment:
```bash
git mv data/mt_bench/model_judgment/gpt-6-luna_pair.jsonl \
       data/mt_bench/model_judgment/gpt-6-luna-antiverbosity_pair.jsonl
python gen_judgment.py --mode pairwise-baseline --judge-model gpt-6-luna \
       --model-list gpt-4 claude-v1 vicuna-13b-v1.3 guanaco-33b alpaca-13b llama-13b --parallel 4
```
(adding `llama-13b` here also covers the Table 11 run below.)

## 2. Needs an OpenAI API key (run on your machine)

The cloud sandbox used for the analysis cannot reach api.openai.com, so these must be run locally.
Always do a `--dry-run` first: same pipeline, fake outputs, zero cost.

```bash
pip install "openai<1"
export OPENAI_API_KEY=sk-...

# 0. Re-judge Luna's 37 failed calls (31 pairwise sides + 6 single).
python -m bias_audit.run_luna_tests rerun-errors

# Paper Table 11 – add the llama-13b pair (big quality gap). 160 rows x 2 orders = 320 calls.
python gen_judgment.py --mode pairwise-baseline --judge-model gpt-6-luna --model-list llama-13b --parallel 4

# Paper Table 4 – math grading: llama-13b vs vicuna-13b on Q111-120, default / CoT / reference prompts. 60 calls.
python -m bias_audit.run_luna_tests t4-math

# Paper Table 3 – verbosity: 23 answers with numbered lists, padded two ways (rephrased list prepended,
# restating summary appended). 46 generation + 92 judge calls.
python -m bias_audit.run_luna_tests t3-verbosity --gen-model <MODEL>

# Paper Table 2 – position bias on near-identical pairs: 2 samples per question at T=0.7. 160 gen + 160 judge.
python -m bias_audit.run_luna_tests t2-position --gen-model <MODEL>

python -m bias_audit.run_luna_tests report
```
Outputs go to `results/api/`. Commit that folder and push.
Then `BIAS_AUDIT_USE_FIXED=1 python -m bias_audit.analyze_existing` regenerates the figures with the re-run errors.

`--gen-model`: the paper used GPT-4 to rephrase (Table 3) and GPT-3.5 sampled twice (Table 2). Use the same models
if your key still has them; otherwise state the substitute in the report. Avoid gpt-6-luna as generator for T3 if
possible (the judge would be grading text it wrote itself).

T4 failure in the paper = "the judge says an incorrect answer is correct", which needs reading the judgment text.
`report` prints automatic proxies; fill `results/api/t4_manual_labels.csv` for the paper's exact metric.
Hand-labelled correctness (vs `reference_answer/gpt-4.jsonl`): llama-13b wrong on all 10; vicuna-13b right on 112, 119.

## 3. Not covered yet
Agreement with human experts (paper Table 5) and self-enhancement (Figure 3) need
`lmsys/mt_bench_human_judgments`; download it and add it to the repo to enable that step.
