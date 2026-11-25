"""Bias tests for gpt-6-luna that need API calls, replicating Zheng et al. (2023).

Run from fastchat/llm_judge with OPENAI_API_KEY set. Every test resumes from
where it stopped (already-finished rows are skipped). Use --dry-run first: it
runs the whole pipeline with fake outputs and costs nothing.

  python -m bias_audit.run_luna_tests rerun-errors            # ~37 calls (31 pairwise sides + 6 single)
  python -m bias_audit.run_luna_tests t4-math                 # 60 calls   (paper Table 4)
  python -m bias_audit.run_luna_tests t3-verbosity --gen-model gpt-6-luna   # 46 gen + 92 judge (paper Table 3)
  python -m bias_audit.run_luna_tests t2-position  --gen-model gpt-3.5-turbo # 160 gen + 160 judge (paper Table 2)
  python -m bias_audit.run_luna_tests report                  # print all metrics, write results/api/report.json

Paper Table 11 (llama pair) uses the repo pipeline directly:
  python gen_judgment.py --mode pairwise-baseline --judge-model gpt-6-luna --model-list llama-13b --parallel 4
"""
import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from bias_audit import luna_client as lc
from bias_audit.common_data import DATA, JUDG, read_jsonl

JUDGE = "gpt-6-luna"
OUT = "bias_audit/results/api"

# Paper Figure 7: chain-of-thought pairwise prompt (not shipped in judge_prompts.jsonl).
COT_SYSTEM = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to "
    "the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be "
    "given assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is "
    "better. You should independently solve the user question step-by-step first. Then compare both assistants' "
    "answers with your answer. Identify and correct any mistakes. Avoid any position biases and ensure that the "
    "order in which the responses were presented does not influence your decision. Do not allow the length of the "
    "responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as "
    "possible. After providing your explanation, output your final verdict by strictly following this format: "
    "\"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
)

# Correctness of turn-1 answers on the 10 math questions (111-120), labelled by hand against
# reference_answer/gpt-4.jsonl. llama-13b is wrong on all 10; vicuna-13b-v1.3 is right on 112 and 119 only.
MATH_CORRECT = {
    111: set(), 112: {"vicuna-13b-v1.3"}, 113: set(), 114: set(), 115: set(),
    116: set(), 117: set(), 118: set(), 119: {"vicuna-13b-v1.3"}, 120: set(),
}


# ------------------------------------------------------------------ helpers
def out_dir():
    d = OUT + ("_dryrun" if lc.DRY_RUN else "")
    os.makedirs(d, exist_ok=True)
    return d


def prompts():
    return {p["name"]: p for p in read_jsonl(os.path.join("data", "judge_prompts.jsonl"))}


def questions():
    return {q["question_id"]: q for q in read_jsonl(os.path.join(DATA, "question.jsonl"))}


def answers(model):
    return {r["question_id"]: r["choices"][0]["turns"] for r in read_jsonl(os.path.join(DATA, "model_answer", f"{model}.jsonl"))}


def ref_answers():
    return {r["question_id"]: r["choices"][0]["turns"] for r in read_jsonl(os.path.join(DATA, "reference_answer", "gpt-4.jsonl"))}


def done_keys(path, keyf):
    if not os.path.exists(path):
        return set(), []
    rows = read_jsonl(path)
    return {keyf(r) for r in rows}, rows


def run_parallel(jobs, fn, path, parallel):
    """jobs: list of dicts; fn(job) -> result dict; appends each result to path as it finishes."""
    if not jobs:
        print(f"nothing to do for {path}")
        return
    print(f"{len(jobs)} jobs -> {path}")
    with ThreadPoolExecutor(parallel) as ex, open(path, "a") as fo:
        for res in ex.map(fn, jobs):
            fo.write(json.dumps(res, ensure_ascii=False) + "\n")
            fo.flush()


def judge_pair(system, template, question, a, b, ref=None):
    kw = {"ref_answer_1": ref} if ref is not None else {}
    user = template.format(question=question, answer_a=a, answer_b=b, **kw)
    j = lc.chat(JUDGE, system, user, temperature=0, max_tokens=2048)
    return lc.parse_pair(j), j


def classify(w1, w2):
    """w1: verdict with X in position A; w2: verdict with X in position B. Returns label for X vs Y."""
    if "error" in (w1, w2):
        return "error"
    x1 = {"A": "X", "B": "Y", "tie": "tie"}[w1]
    x2 = {"A": "Y", "B": "X", "tie": "tie"}[w2]
    if x1 == x2:
        return "consistent:" + x1
    if w1 == "A" and w2 == "A":
        return "bias_first"
    if w1 == "B" and w2 == "B":
        return "bias_second"
    return "split_tie"


# ------------------------------------------------------------------ rerun errors
def cmd_rerun_errors(args):
    P = prompts()
    d = out_dir()
    # pairwise
    rows = read_jsonl(os.path.join(JUDG, f"{JUDGE}_pair.jsonl"))
    fixed = []
    n_calls = 0
    for r in rows:
        r = dict(r)
        sys_prompt = P[r["judge"][1]]["system_prompt"]
        for g in ("g1", "g2"):
            if r[f"{g}_winner"] == "error":
                n_calls += 1
                j = lc.chat(JUDGE, sys_prompt, r[f"{g}_user_prompt"], temperature=0, max_tokens=2048)
                w = lc.parse_pair(j)
                # stored winner is relative to model_1/model_2: g1 has model_1 as A, g2 has model_1 as B
                if g == "g1":
                    r[f"{g}_winner"] = {"A": "model_1", "B": "model_2", "tie": "tie", "error": "error"}[w]
                else:
                    r[f"{g}_winner"] = {"A": "model_2", "B": "model_1", "tie": "tie", "error": "error"}[w]
                r[f"{g}_judgment"] = j
                r["rerun"] = True
        fixed.append(r)
    with open(os.path.join(d, f"{JUDGE}_pair_fixed.jsonl"), "w") as fo:
        for r in fixed:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    # single
    rows = read_jsonl(os.path.join(JUDG, f"{JUDGE}_single.jsonl"))
    fixed = []
    for r in rows:
        r = dict(r)
        if r["score"] == -1:
            n_calls += 1
            j = lc.chat(JUDGE, P[r["judge"][1]]["system_prompt"], r["user_prompt"], temperature=0, max_tokens=2048)
            r["judgment"], r["score"], r["rerun"] = j, lc.parse_single(j), True
        fixed.append(r)
    with open(os.path.join(d, f"{JUDGE}_single_fixed.jsonl"), "w") as fo:
        for r in fixed:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    still = sum(1 for r in read_jsonl(os.path.join(d, f"{JUDGE}_pair_fixed.jsonl"))
                if "error" in (r["g1_winner"], r["g2_winner"]))
    print(f"rerun calls: {n_calls}; pair rows still in error: {still}. Files in {d}/")


# ------------------------------------------------------------------ T4 math grading (paper Table 4)
def cmd_t4(args):
    P, Q, R = prompts(), questions(), ref_answers()
    A = {"llama-13b": answers("llama-13b"), "vicuna-13b-v1.3": answers("vicuna-13b-v1.3")}
    variants = {
        "default": (P["pair-v2"]["system_prompt"], P["pair-v2"]["prompt_template"], False),
        "cot": (COT_SYSTEM, P["pair-v2"]["prompt_template"], False),
        "reference": (P["pair-math-v1"]["system_prompt"], P["pair-math-v1"]["prompt_template"], True),
    }
    path = os.path.join(out_dir(), "t4_math.jsonl")
    keys, _ = done_keys(path, lambda r: (r["question_id"], r["prompt"], r["order"]))
    jobs = [dict(q=q, v=v, order=o) for q in range(111, 121) for v in variants for o in ("llama_first", "vicuna_first")
            if (q, v, o) not in keys]

    def fn(job):
        sys_p, tmpl, use_ref = variants[job["v"]]
        m_a, m_b = ("llama-13b", "vicuna-13b-v1.3") if job["order"] == "llama_first" else ("vicuna-13b-v1.3", "llama-13b")
        w, j = judge_pair(sys_p, tmpl, Q[job["q"]]["turns"][0], A[m_a][job["q"]][0], A[m_b][job["q"]][0],
                          ref=R[job["q"]][0] if use_ref else None)
        winner = {"A": m_a, "B": m_b, "tie": "tie", "error": "error"}[w]
        return {"question_id": job["q"], "prompt": job["v"], "order": job["order"], "model_a": m_a, "model_b": m_b,
                "winner": winner, "judgment": j}

    run_parallel(jobs, fn, path, args.parallel)
    # manual-labelling sheet: does the judgment claim an incorrect answer is correct? (paper's failure definition)
    rows = read_jsonl(path)
    with open(os.path.join(out_dir(), "t4_manual_labels.csv"), "w") as fo:
        fo.write("question_id,prompt,order,winner,winner_is_correct,says_incorrect_is_correct(fill 0/1)\n")
        for r in sorted(rows, key=lambda r: (r["question_id"], r["prompt"], r["order"])):
            ok = r["winner"] in MATH_CORRECT[r["question_id"]]
            fo.write(f'{r["question_id"]},{r["prompt"]},{r["order"]},{r["winner"]},{int(ok)},\n')


# ------------------------------------------------------------------ T3 verbosity / repetitive list (paper Table 3)
NUM = re.compile(r"^(\s*)(\d+)([.)])\s+(.*)$")
PAD_MODELS = ["gpt-4", "gpt-3.5-turbo", "claude-v1", "vicuna-13b-v1.3", "guanaco-33b", "alpaca-13b"]
PAD_CATS = {"writing", "roleplay", "stem", "humanities", "extraction"}


def find_list(text):
    """Return (start, end) line indices of the first numbered list with >= 3 items numbered 1..n."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = NUM.match(lines[i])
        if m and m.group(2) == "1":
            expect, last, j = 2, i, i + 1
            while j < len(lines):
                mj = NUM.match(lines[j])
                if mj and int(mj.group(2)) == expect:
                    expect, last = expect + 1, j
                elif mj or (lines[j].strip() == "" and j + 1 < len(lines) and not NUM.match(lines[j + 1])
                             and not lines[j + 1].startswith((" ", "\t", "-", "*"))):
                    break
                j += 1
            if expect - 1 >= 3:
                end = last + 1
                while end < len(lines) and lines[end].strip() and not NUM.match(lines[end]):
                    end += 1  # continuation lines of the last item
                return i, end, expect - 1
        i += 1
    return None


def select_pad_items(n=23, seed=0):
    Q = questions()
    cands = {}
    for m in PAD_MODELS:
        for qid, turns in answers(m).items():
            if Q[qid]["category"] in PAD_CATS and find_list(turns[0]):
                cands.setdefault(qid, []).append(m)
    rng = random.Random(seed)
    qids = sorted(cands)
    rng.shuffle(qids)
    return [(q, rng.choice(sorted(cands[q]))) for q in sorted(qids[:n])]


def renumber(block_lines, start):
    out, k = [], start
    for ln in block_lines:
        m = NUM.match(ln)
        if m:
            out.append(f"{m.group(1)}{k}{m.group(3)} {m.group(4)}")
            k += 1
        else:
            out.append(ln)
    return out


def cmd_t3(args):
    if not args.gen_model:
        raise SystemExit("t3-verbosity needs --gen-model (the model that rephrases/summarises; paper used GPT-4)")
    P, Q = prompts(), questions()
    items = select_pad_items()
    d = out_dir()
    pad_path = os.path.join(d, "t3_padded_answers.jsonl")
    keys, _ = done_keys(pad_path, lambda r: (r["question_id"], r["model"], r["pad_type"]))

    def make_pad(job):
        q, m, t = job["q"], job["m"], job["t"]
        orig = answers(m)[q][0]
        s, e, n = find_list(orig)
        lines = orig.split("\n")
        block = lines[s:e]
        if t == "rephrase":
            req = ("Rephrase each item of the following numbered list in different words. Do not add any new "
                   "information and do not remove any. Keep exactly the same number of items. Return only the list, "
                   "one item per line, numbered 1., 2., 3., ...\n\n" + "\n".join(block))
            reph = lc.chat(args.gen_model, "You are a helpful assistant.", req, temperature=0, max_tokens=1024)
            reph_lines = [ln for ln in reph.split("\n") if NUM.match(ln)]
            padded = "\n".join(lines[:s] + renumber(reph_lines, 1) + renumber(block, len(reph_lines) + 1) + lines[e:])
        else:
            req = ("Write a short closing summary paragraph (3-4 sentences) that restates the main points of the "
                   "answer below WITHOUT adding any new information. Return only the paragraph.\n\n" + orig)
            summ = lc.chat(args.gen_model, "You are a helpful assistant.", req, temperature=0, max_tokens=512)
            padded = orig.rstrip() + "\n\n" + summ.strip()
        res = {"question_id": q, "model": m, "pad_type": t, "n_items": n, "original": orig, "padded": padded,
               "gen_model": args.gen_model}
        if t == "rephrase":
            res["rephrased_items"] = len(reph_lines)
            if len(reph_lines) != n:
                print(f"WARNING q{q} {m}: rephrased {len(reph_lines)} items, original {n} -- check by hand")
        return res

    jobs = [dict(q=q, m=m, t=t) for q, m in items for t in ("rephrase", "summary") if (q, m, t) not in keys]
    run_parallel(jobs, make_pad, pad_path, args.parallel)

    pads = read_jsonl(pad_path)
    jpath = os.path.join(d, "t3_verbosity_judgments.jsonl")
    keys, _ = done_keys(jpath, lambda r: (r["question_id"], r["model"], r["pad_type"], r["order"]))
    sys_p, tmpl = P["pair-v2"]["system_prompt"], P["pair-v2"]["prompt_template"]

    def judge(job):
        p = job["p"]
        a, b = (p["original"], p["padded"]) if job["o"] == "orig_first" else (p["padded"], p["original"])
        w, j = judge_pair(sys_p, tmpl, Q[p["question_id"]]["turns"][0], a, b)
        win = {"A": "original" if job["o"] == "orig_first" else "padded",
               "B": "padded" if job["o"] == "orig_first" else "original", "tie": "tie", "error": "error"}[w]
        return {"question_id": p["question_id"], "model": p["model"], "pad_type": p["pad_type"], "order": job["o"],
                "winner": win, "judgment": j}

    jobs = [dict(p=p, o=o) for p in pads for o in ("orig_first", "padded_first")
            if (p["question_id"], p["model"], p["pad_type"], o) not in keys]
    run_parallel(jobs, judge, jpath, args.parallel)


# ------------------------------------------------------------------ T2 position, near-identical pairs (paper Table 2)
def cmd_t2(args):
    if not args.gen_model:
        raise SystemExit("t2-position needs --gen-model (paper sampled GPT-3.5 twice at temperature 0.7)")
    P, Q = prompts(), questions()
    d = out_dir()
    gpath = os.path.join(d, "t2_samples.jsonl")
    keys, _ = done_keys(gpath, lambda r: (r["question_id"], r["sample"]))
    jobs = [dict(q=q, s=s) for q in sorted(Q) for s in (0, 1) if (q, s) not in keys]

    def gen(job):
        txt = lc.chat(args.gen_model, "You are a helpful assistant.", Q[job["q"]]["turns"][0], temperature=0.7,
                      max_tokens=1024)
        return {"question_id": job["q"], "sample": job["s"], "answer": txt, "gen_model": args.gen_model}

    run_parallel(jobs, gen, gpath, args.parallel)
    S = {}
    for r in read_jsonl(gpath):
        S[(r["question_id"], r["sample"])] = r["answer"]
    jpath = os.path.join(d, "t2_position_judgments.jsonl")
    keys, _ = done_keys(jpath, lambda r: (r["question_id"], r["order"]))
    sys_p, tmpl = P["pair-v2"]["system_prompt"], P["pair-v2"]["prompt_template"]

    def judge(job):
        q = job["q"]
        a, b = (S[(q, 0)], S[(q, 1)]) if job["o"] == "s0_first" else (S[(q, 1)], S[(q, 0)])
        w, j = judge_pair(sys_p, tmpl, Q[q]["turns"][0], a, b)
        return {"question_id": q, "category": Q[q]["category"], "order": job["o"], "verdict_position": w,
                "judgment": j}

    jobs = [dict(q=q, o=o) for q in sorted(Q) for o in ("s0_first", "s1_first")
            if (q, 0) in S and (q, 1) in S and (q, o) not in keys]
    run_parallel(jobs, judge, jpath, args.parallel)


# ------------------------------------------------------------------ report
def pct(n, d):
    return round(100 * n / d, 1) if d else None


def cmd_report(args):
    d = out_dir()
    rep = {}
    p = os.path.join(d, "t2_position_judgments.jsonl")
    if os.path.exists(p):
        rows = read_jsonl(p)
        by = {}
        for r in rows:
            by.setdefault(r["question_id"], {})[r["order"]] = r["verdict_position"]
        labels = [classify(v["s0_first"], v["s1_first"]) for v in by.values() if len(v) == 2]
        n = len(labels)
        rep["T2_position_near_identical"] = {
            "n_pairs": n,
            "consistent_%": pct(sum(l.startswith("consistent") for l in labels), n),
            "biased_first_%": pct(labels.count("bias_first"), n),
            "biased_second_%": pct(labels.count("bias_second"), n),
            "split_tie_%": pct(labels.count("split_tie"), n),
            "error_%": pct(labels.count("error"), n),
            "paper_gpt4": {"consistent_%": 65.0, "biased_first_%": 30.0, "biased_second_%": 5.0, "error_%": 0.0},
        }
    p = os.path.join(d, "t3_verbosity_judgments.jsonl")
    if os.path.exists(p):
        rows = read_jsonl(p)
        rep["T3_verbosity_repetitive"] = {}
        for t in ("rephrase", "summary"):
            by = {}
            for r in rows:
                if r["pad_type"] == t:
                    by.setdefault((r["question_id"], r["model"]), {})[r["order"]] = r["winner"]
            full = [v for v in by.values() if len(v) == 2]
            n = len(full)
            rep["T3_verbosity_repetitive"][t] = {
                "n_answers": n,
                "fail_both_orders_%": pct(sum(v["orig_first"] == v["padded_first"] == "padded" for v in full), n),
                "fail_any_order_%": pct(sum("padded" in v.values() for v in full), n),
                "original_wins_both_%": pct(sum(v["orig_first"] == v["padded_first"] == "original" for v in full), n),
            }
        rep["T3_verbosity_repetitive"]["paper_failure_%"] = {"claude-v1": 91.3, "gpt-3.5": 91.3, "gpt-4": 8.7}
    p = os.path.join(d, "t4_math.jsonl")
    if os.path.exists(p):
        rows = read_jsonl(p)
        res = {}
        for v in ("default", "cot", "reference"):
            sub = [r for r in rows if r["prompt"] == v]
            one_correct = [r for r in sub if MATH_CORRECT[r["question_id"]]]
            both_wrong = [r for r in sub if not MATH_CORRECT[r["question_id"]]]
            res[v] = {
                "n": len(sub),
                "picked_correct_when_one_correct": f"{sum(r['winner'] in MATH_CORRECT[r['question_id']] for r in one_correct)}/{len(one_correct)}",
                "declared_winner_when_both_wrong": f"{sum(r['winner'] not in ('tie', 'error') for r in both_wrong)}/{len(both_wrong)}",
                "errors": sum(r["winner"] == "error" for r in sub),
            }
        res["note"] = ("Paper failure = judge says an incorrect answer is correct (GPT-4: default 14/20, CoT 6/20, "
                       "reference 3/20). Fill t4_manual_labels.csv to compute the same metric; the counts above are "
                       "automatic proxies from the verdict only.")
        rep["T4_math_grading"] = res
    with open(os.path.join(d, "report.json"), "w") as fo:
        json.dump(rep, fo, indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rerun-errors", "t2-position", "t3-verbosity", "t4-math", "report", "pad-preview"])
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="no API calls; fake outputs into results/api_dryrun/")
    args = ap.parse_args()
    lc.DRY_RUN = args.dry_run
    if not args.dry_run and args.cmd not in ("report", "pad-preview") and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first (or use --dry-run).")
    if args.cmd == "pad-preview":
        for q, m in select_pad_items():
            print(q, m, find_list(answers(m)[q][0])[2], "items")
        return
    {"rerun-errors": cmd_rerun_errors, "t2-position": cmd_t2, "t3-verbosity": cmd_t3, "t4-math": cmd_t4,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
