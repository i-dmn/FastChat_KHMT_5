"""Shared loaders for the bias-audit analysis.

All paths are relative to fastchat/llm_judge. Run scripts from that directory:
    cd fastchat/llm_judge
    python -m bias_audit.analyze_existing
"""
import json
import os

import numpy as np
import pandas as pd

DATA = "data/mt_bench"
JUDG = os.path.join(DATA, "model_judgment")
MODELS = ["gpt-4", "claude-v1", "vicuna-13b-v1.3", "guanaco-33b", "alpaca-13b"]
BASELINE = "gpt-3.5-turbo"
SHORT = {
    "gpt-4": "gpt-4",
    "claude-v1": "claude-v1",
    "vicuna-13b-v1.3": "vicuna-13b",
    "guanaco-33b": "guanaco-33b",
    "alpaca-13b": "alpaca-13b",
    "gpt-3.5-turbo": "gpt-3.5",
    "llama-13b": "llama-13b",
}
CAT_ORDER = ["math", "reasoning", "coding", "extraction",
             "roleplay", "writing", "stem", "humanities"]
CAT_LABEL = {c: c.capitalize() if c != "stem" else "STEM" for c in CAT_ORDER}
OBJECTIVE = ["math", "reasoning", "coding", "extraction"]
JUDGES = {"gpt-4": "GPT-4 judge", "gpt-6-luna": "gpt-6-luna judge"}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def question_categories():
    return {q["question_id"]: q["category"] for q in read_jsonl(os.path.join(DATA, "question.jsonl"))}


def load_answers(models):
    """Return {(model, question_id, turn): answer_text}."""
    out = {}
    for m in models:
        for r in read_jsonl(os.path.join(DATA, "model_answer", f"{m}.jsonl")):
            for t, txt in enumerate(r["choices"][0]["turns"], start=1):
                out[(m, r["question_id"], t)] = txt
    return out


USE_FIXED = os.environ.get("BIAS_AUDIT_USE_FIXED") == "1"
FIXED_DIR = "bias_audit/results/api"


def judgment_path(judge, kind):
    """Original repo file, or the re-run version (errors re-judged) if BIAS_AUDIT_USE_FIXED=1 and it exists."""
    fixed = os.path.join(FIXED_DIR, f"{judge}_{kind}_fixed.jsonl")
    if USE_FIXED and os.path.exists(fixed):
        return fixed
    return os.path.join(JUDG, f"{judge}_{kind}.jsonl")


def load_single(judge):
    df = pd.DataFrame(read_jsonl(judgment_path(judge, "single")))
    df = df[df.model.isin(MODELS + [BASELINE])].copy()
    df["cat"] = df.question_id.map(question_categories())
    df["prompt_type"] = df.judge.apply(lambda j: j[1])
    df["valid"] = df.score != -1
    return df


def _final(a, b):
    """Paper's conservative rule: win only if both orders agree, else tie."""
    if "error" in (a, b):
        return "error"
    if a == b:
        return {"model_1": "win", "model_2": "loss", "tie": "tie"}[a]
    return "tie"


def _consistency(a, b):
    if "error" in (a, b):
        return "error"
    if a == b:
        return "consistent"
    # g1: model_1 in position A. g2: model_1 in position B (order swapped).
    if a == "model_1" and b == "model_2":
        return "bias_first"   # position A won both times
    if a == "model_2" and b == "model_1":
        return "bias_second"  # position B won both times
    return "split_tie"        # one order tie, other order a winner


def load_pair(judge):
    df = pd.DataFrame(read_jsonl(judgment_path(judge, "pair")))
    df = df[df.model_2 == BASELINE].copy()
    df["cat"] = df.question_id.map(question_categories())
    df["prompt_type"] = df.judge.apply(lambda j: j[1])
    df["final"] = [_final(a, b) for a, b in zip(df.g1_winner, df.g2_winner)]
    df["cons"] = [_consistency(a, b) for a, b in zip(df.g1_winner, df.g2_winner)]
    df["s"] = df["final"].map({"win": 1.0, "tie": 0.5, "loss": 0.0})
    return df


def bootstrap_mean(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def bootstrap_by_question(df, stat, n=2000, seed=0):
    """Cluster bootstrap: resample question_ids, recompute stat(df_sample)."""
    rng = np.random.default_rng(seed)
    qids = df.question_id.unique()
    groups = {q: g for q, g in df.groupby("question_id")}
    vals = []
    for _ in range(n):
        sample = rng.choice(qids, len(qids), replace=True)
        vals.append(stat(pd.concat([groups[q] for q in sample], ignore_index=True)))
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return stat(df), np.percentile(vals, 2.5), np.percentile(vals, 97.5)
