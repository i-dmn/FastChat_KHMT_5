# Next steps: finish the Luna bias audit (run on a machine with OPENAI_API_KEY)

Work from `fastchat/llm_judge/` on branch `bias-audit-luna`. Every API step is resumable: if a run stops, rerun the same command.

## 0. Environment
```
pip install -e "../..[llm_judge]" torch pandas numpy scipy matplotlib huggingface_hub
```
`OPENAI_API_KEY` must be set in the environment. Never commit it.

## 1. Check whether gpt-6-luna returns logprobs (decides improvement P1)
```
python -c "import openai; r=openai.ChatCompletion.create(model='gpt-6-luna',messages=[{'role':'user','content':'Reply A or B'}],max_completion_tokens=5,reasoning_effort='none',logprobs=True,top_logprobs=5); print(r['choices'][0].get('logprobs'))"
```
Write the output to `bias_audit/results/api/logprobs_check.txt`.

## 2. Re-judge failed rows of the EXISTING pairwise file (must run before step 3)
```
python -m bias_audit.run_luna_tests rerun-errors
```

## 3. Clean pairwise re-run with the paper's original prompt
The existing `gpt-6-luna_pair.jsonl` was produced while the ANTI-VERBOSITY prompt (commit 4439e13) was in place. Keep it under a new name, then re-run:
```
git mv data/mt_bench/model_judgment/gpt-6-luna_pair.jsonl data/mt_bench/model_judgment/gpt-6-luna-antiverbosity_pair.jsonl
python gen_judgment.py --mode pairwise-baseline --judge-model gpt-6-luna --model-list gpt-4 claude-v1 vicuna-13b-v1.3 guanaco-33b alpaca-13b llama-13b --parallel 4 --yes
```

## 4. Replicate paper Tables 4, 3, 2
GEN_MODEL = an OpenAI model other than gpt-6-luna (paper used gpt-3.5-turbo / gpt-4).
```
python -m bias_audit.run_luna_tests t4-math
python -m bias_audit.run_luna_tests t3-verbosity --gen-model GEN_MODEL
python -m bias_audit.run_luna_tests t2-position --gen-model GEN_MODEL
python -m bias_audit.run_luna_tests report
```
T4 also writes `results/api/t4_manual_labels.csv`; the last column must be filled by hand (0/1).

## 5. Human expert votes (paper Table 5)
```
huggingface-cli download lmsys/mt_bench_human_judgments --repo-type dataset --local-dir data/mt_bench/human_judgments
```

## 6. Regenerate analysis and push
```
BIAS_AUDIT_USE_FIXED=1 python -m bias_audit.analyze_existing
git add -f bias_audit/results data/mt_bench/model_judgment data/mt_bench/human_judgments
git commit -m "Luna re-runs, T2-T4, human judgments" && git push
```

## Still open after this
- Luna must judge the pairs that the human experts voted on (for Table 5 / E1-E4 / V2). Script not written yet: needs the downloaded data to build the pair list. `compute_agreement.py:68` only accepts judges named `gpt4*` and needs extending.
- Improvement P2 (single-answer tie-break for inconsistent pairs) needs no API calls.
- Questions for the person who ran the Luna jobs (kaneblakes): which prompt the pairwise run used; whether the single run (Nov 10, 18:18) used `reasoning_effort="none"`; why temperature is not sent for gpt-6 models.
