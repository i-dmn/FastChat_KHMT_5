"""Bias audit of GPT-4 vs gpt-6-luna from the judgments already in the repo.

No API calls. Reproduces every number and chart used in the slides, plus the
bias tests that can be done on existing data:
  - position bias by model pair and category (paper Tables 10-11 style)
  - length (verbosity) preference on real MT-Bench answers

Usage (from fastchat/llm_judge):
    python -m bias_audit.analyze_existing
Outputs:
    bias_audit/figures/*.png   slide-ready charts (16:9)
    bias_audit/results/summary.json
    bias_audit/results/*.csv
"""
import json
import os
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from bias_audit.common_data import (BASELINE, CAT_LABEL, CAT_ORDER, MODELS,
                                    OBJECTIVE, SHORT, bootstrap_by_question,
                                    bootstrap_mean, load_answers, load_pair,
                                    load_single)

OUT_FIG = "bias_audit/figures"
OUT_RES = "bias_audit/results"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold"})
CG, CL = "#2E5BBA", "#E8833A"
WIN, TIE, LOSS = "#3B8EA5", "#BDBDBD", "#D1495B"
LAB = [SHORT[m] for m in MODELS]
X = np.arange(len(MODELS))
SUMMARY = {}


def fig(title, sub=None):
    f = plt.figure(figsize=(13.33, 7.5))
    f.suptitle(title, x=0.04, y=0.965, ha="left", fontsize=21, fontweight="bold")
    if sub:
        f.text(0.04, 0.895, sub, ha="left", fontsize=14, color="#444")
    return f


def foot(f, t):
    f.text(0.04, 0.012, "\n".join(textwrap.wrap("Nguồn: " + t, 158)),
           ha="left", va="bottom", fontsize=9.5, color="#666", linespacing=1.25)


def save(f, name):
    f.savefig(os.path.join(OUT_FIG, name), dpi=200)
    plt.close(f)


def words(t):
    return len(re.findall(r"\S+", t))


def list_items(t):
    return len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+", t))


# ---------------------------------------------------------------- load
gs = load_single("gpt-4")
ls = load_single("gpt-6-luna")
gs5, ls5 = gs[gs.model.isin(MODELS)], ls[ls.model.isin(MODELS)]
lsv = ls5[ls5.valid]
gp_all, lp_all = load_pair("gpt-4"), load_pair("gpt-6-luna")
gp, lp = gp_all[gp_all.model_1.isin(MODELS)], lp_all[lp_all.model_1.isin(MODELS)]
ans = load_answers(MODELS + [BASELINE])
JUDGE_DATA = (("gpt-4", gs5, gp, CG), ("gpt-6-luna", lsv, lp, CL))

# ---------------------------------------------------------------- 01 single means
f = fig("Điểm single: cùng hạng đầu–cuối, vicuna và guanaco đổi chỗ",
        "Điểm trung bình (thang 1–10), 2 turn gộp; thanh sai số = 95% CI (bootstrap theo câu hỏi)")
ax = f.add_axes([0.07, 0.17, 0.90, 0.66])
w = 0.38
SUMMARY["single_mean"] = {}
for i, (jn, d, _, col) in enumerate(JUDGE_DATA):
    m = np.array([bootstrap_mean(d[d.model == mo].groupby("question_id").score.mean().values)
                  for mo in MODELS])
    SUMMARY["single_mean"][jn] = dict(zip(MODELS, m[:, 0].round(3).tolist()))
    ax.bar(X + (i - .5) * w, m[:, 0], w, color=col, label=f"{jn} judge",
           yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], capsize=4)
    for xi, v in zip(X + (i - .5) * w, m[:, 0]):
        ax.text(xi, v + 0.45, f"{v:.2f}", ha="center", fontsize=13, fontweight="bold")
ax.set_xticks(X)
ax.set_xticklabels(LAB)
ax.set_ylim(0, 10.5)
ax.set_ylabel("Điểm trung bình")
ax.legend(frameon=False, loc="upper right")
foot(f, "model_judgment/gpt-4_single.jsonl (phán quyết tác giả công bố, 6/2023) và gpt-6-luna_single.jsonl "
        "(nhóm chạy, 11/2026); loại 6 dòng score = -1 của Luna. GPT-4 trùng Bảng 8 của paper (8.99 / 6.39 / 4.53).")
save(f, "01_single_score_by_model.png")

# ---------------------------------------------------------------- 02 turn effect single
f = fig("Turn 2: GPT-4 judge phạt theo model, Luna phạt đồng đều",
        "Điểm trung bình theo turn (single-answer); Δ = turn 2 − turn 1")
SUMMARY["single_turn"] = {}
for k, (jn, d, _, col) in enumerate(JUDGE_DATA):
    ax = f.add_axes([0.07 + k * 0.48, 0.17, 0.43, 0.62])
    t = d.groupby(["model", "turn"]).score.mean().unstack().loc[MODELS]
    SUMMARY["single_turn"][jn] = t.round(3).to_dict()
    ax.bar(X - .2, t[1], 0.38, color=col, alpha=0.55, label="Turn 1")
    ax.bar(X + .2, t[2], 0.38, color=col, label="Turn 2")
    for xi, mo in zip(X, MODELS):
        ax.text(xi, max(t.loc[mo]) + 0.55, f"Δ {t.loc[mo, 2] - t.loc[mo, 1]:+.2f}",
                ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(X)
    ax.set_xticklabels(LAB, rotation=20, fontsize=12)
    ax.set_ylim(0, 11)
    ax.set_title(f"{jn} judge", color=col, fontsize=17)
    ax.legend(frameon=False, fontsize=12, loc="upper right")
foot(f, "*_single.jsonl, cột score theo cột turn (loại score = -1). Wilcoxon cặp theo câu hỏi: Luna giảm có ý nghĩa "
        "ở cả 5 model (p < 0.001); GPT-4 judge: gpt-4 +0.07 (không có ý nghĩa).")
save(f, "02_turn_effect_single.png")

# ---------------------------------------------------------------- 03 category gap
mm = gs5[["question_id", "model", "turn", "score", "cat"]].merge(
    lsv[["question_id", "model", "turn", "score"]], on=["question_id", "model", "turn"], suffixes=("_g", "_l"))
SUMMARY["single_pearson_all"] = round(stats.pearsonr(mm.score_g, mm.score_l)[0], 3)
cat_rows = []
f = fig("Hai judge lệch nhiều nhất ở câu hỏi mở (open-ended)",
        "Điểm trung bình theo category (5 model, 2 turn); r = tương quan Pearson giữa hai judge trên từng mục chấm")
ax = f.add_axes([0.17, 0.14, 0.62, 0.70])
ys = np.arange(len(CAT_ORDER))[::-1]
for y, c in zip(ys, CAT_ORDER):
    x_ = mm[mm.cat == c]
    g, l = x_.score_g.mean(), x_.score_l.mean()
    r = stats.pearsonr(x_.score_g, x_.score_l)[0]
    cat_rows.append(dict(cat=c, gpt4=round(g, 2), luna=round(l, 2), diff=round(g - l, 2), r=round(r, 2)))
    ax.plot([l, g], [y, y], color="#999", lw=3, zorder=1)
    ax.scatter([g], [y], s=190, color=CG, zorder=2)
    ax.scatter([l], [y], s=190, color=CL, zorder=2)
    ax.text(10.4, y, f"Δ {g - l:+.1f}   r = {r:.2f}".replace("-", "−"), va="center", fontsize=13, fontweight="bold")
pd.DataFrame(cat_rows).to_csv(os.path.join(OUT_RES, "single_by_category.csv"), index=False)
ax.set_yticks(ys)
ax.set_yticklabels([CAT_LABEL[c] for c in CAT_ORDER], fontsize=15)
ax.set_xlim(2, 10.2)
ax.set_xlabel("Điểm trung bình")
ax.axhline(3.5, color="#bbb", ls="--", lw=1.2)
ax.text(2.05, 3.62, "Có đáp án / kiểm chứng được", fontsize=11, color="#777")
ax.text(2.05, 3.38, "Câu hỏi mở", fontsize=11, color="#777", va="top")
ax.legend(handles=[Patch(color=CG, label="GPT-4 judge"), Patch(color=CL, label="gpt-6-luna judge")],
          frameon=False, fontsize=13, loc="upper right")
foot(f, "*_single.jsonl ghép theo (question_id, model, turn); category theo question.jsonl. "
        "Math/Reasoning/Coding dùng prompt có đáp án tham chiếu (single-math-v1).")
save(f, "03_category_gap.png")

# ---------------------------------------------------------------- 04 distribution
f = fig("GPT-4 judge chấm hai cực, Luna rải đều ở giữa",
        "Tỉ lệ các mức điểm 1–10 trên toàn bộ mục chấm single-answer (5 model × 160)")
ax = f.add_axes([0.07, 0.17, 0.90, 0.66])
xs = np.arange(1, 11)
for i, (jn, d, _, col) in enumerate(JUDGE_DATA):
    s = np.floor(d.score + 0.5).astype(int).clip(1, 10)
    pct = s.value_counts(normalize=True).reindex(xs).fillna(0) * 100
    SUMMARY.setdefault("score_10_pct", {})[jn] = round(pct[10], 1)
    ax.bar(xs + (i - .5) * 0.38, pct, 0.38, color=col,
           label=f"{jn} judge (TB {d.score.mean():.2f}, SD {d.score.std():.2f})")
ax.set_xticks(xs)
ax.set_xlabel("Điểm")
ax.set_ylabel("% mục chấm")
ax.legend(frameon=False, loc="upper left")
foot(f, "*_single.jsonl, cột score (điểm .5 làm tròn lên; loại score = -1).")
save(f, "04_score_distribution.png")


# ---------------------------------------------------------------- 05 pair outcomes
def outcomes(d):
    v = d[d["final"] != "error"]
    return v.groupby("model_1")["final"].value_counts().unstack().fillna(0).loc[MODELS][["win", "tie", "loss"]]


f = fig("Pairwise: cùng hạng đầu–cuối, vicuna và guanaco đổi chỗ",
        "Kết quả cuối của mỗi model so với baseline gpt-3.5-turbo; bất nhất khi đổi vị trí = hòa (quy tắc của paper)")
SUMMARY["pair_winrate"] = {}
for k, (jn, _, d, col) in enumerate(JUDGE_DATA):
    ax = f.add_axes([0.14 + k * 0.43, 0.15, 0.38, 0.64])
    t = outcomes(d)
    n = t.sum(axis=1)
    pc = t.div(n, axis=0) * 100
    yy = np.arange(5)[::-1]
    left = np.zeros(5)
    for key, cc in (("win", WIN), ("tie", TIE), ("loss", LOSS)):
        ax.barh(yy, pc[key], left=left, color=cc, height=0.62)
        for yi, l_, v_ in zip(yy, left, pc[key]):
            if v_ >= 7:
                ax.text(l_ + v_ / 2, yi, f"{v_:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if key != "tie" else "#333")
        left += pc[key].values
    adj = (t.win + .5 * t.tie) / n
    SUMMARY["pair_winrate"][jn] = adj.round(3).to_dict()
    for yi, mo in zip(yy, MODELS):
        ax.text(101, yi, f"{adj[mo]:.2f}", va="center", fontsize=13, fontweight="bold")
    ax.text(101, 4.62, "win-rate*", fontsize=11, color="#555")
    ax.set_xlim(0, 112)
    ax.set_ylim(-.6, 4.9)
    ax.set_yticks(yy)
    ax.set_yticklabels(LAB if k == 0 else [""] * 5)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0%", "50%", "100%"])
    ax.set_title(f"{jn} judge  (n = {int(n.sum())} lượt)", color=col, fontsize=16)
f.legend(handles=[Patch(color=WIN, label="Model thắng"), Patch(color=TIE, label="Hòa / bất nhất"),
                  Patch(color=LOSS, label="gpt-3.5-turbo thắng")],
         loc="upper right", bbox_to_anchor=(0.985, 0.905), ncol=3, frameon=False, fontsize=12)
foot(f, "*_pair.jsonl, cột g1_winner/g2_winner (hai thứ tự A/B); loại dòng 'error' (Luna: 24). "
        "*win-rate = (thắng + 0.5 × hòa)/n, cách tính của FastChat.")
save(f, "05_pairwise_outcomes.png")

# ---------------------------------------------------------------- 06 pair turn effect
f = fig("Pairwise: GPT-4 judge thấy hiệu ứng turn 2, Luna thì không",
        "Thay đổi điểm pairwise so với gpt-3.5-turbo: turn 2 − turn 1 (thắng = 1, hòa = 0.5, thua = 0); thanh = 95% CI")
ax = f.add_axes([0.07, 0.17, 0.90, 0.66])
for i, (jn, _, d, col) in enumerate(JUDGE_DATA):
    v = d[d["final"] != "error"]
    res = []
    for mo in MODELS:
        p = v[v.model_1 == mo].pivot_table(index="question_id", columns="turn", values="s").dropna()
        res.append(bootstrap_mean((p[2] - p[1]).values))
    r = np.array(res)
    xx = X + (i - .5) * 0.36
    ax.errorbar(xx, r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], fmt="o", ms=12, color=col,
                capsize=6, lw=2.2, label=f"{jn} judge")
    for xi, m_, lo, hi in zip(xx, r[:, 0], r[:, 1], r[:, 2]):
        ax.text(xi + 0.07, m_, f"{m_:+.2f}" + (" *" if (lo > 0 or hi < 0) else ""), va="center", fontsize=12)
ax.axhline(0, color="#555", lw=1.2)
ax.set_xticks(X)
ax.set_xticklabels(LAB)
ax.set_ylabel("Δ điểm pairwise (turn 2 − turn 1)")
ax.legend(frameon=False, loc="lower left")
ax.set_ylim(-0.4, 0.3)
foot(f, "*_pair.jsonl (g1_winner, g2_winner, turn), ghép cặp theo câu hỏi. * = CI không chứa 0. Pairwise so với cùng "
        "baseline nên triệt tiêu mức phạt đồng đều của judge ⇒ mức giảm ~1 điểm của Luna ở single nhiều khả năng là "
        "dịch chuyển thang điểm (suy luận, chưa kiểm chứng trực tiếp).")
save(f, "06_pair_turn_effect.png")

# ---------------------------------------------------------------- 07 position consistency
ORDER7 = [("consistent", "Nhất quán", WIN), ("bias_first", "Luôn chọn câu ở vị trí đầu", "#F2A541"),
          ("bias_second", "Luôn chọn câu ở vị trí sau", "#8E6BBF"),
          ("split_tie", "Lệch giữa hòa và thắng", "#BDBDBD"), ("error", "Lỗi (phán quyết rỗng)", "#D1495B")]
f = fig("Thiên kiến vị trí trên dữ liệu thật: Luna nhất quán hơn GPT-4",
        "Mỗi cặp được chấm 2 lần với thứ tự A/B đảo nhau; nhất quán = cùng kết luận ở cả hai thứ tự")
ax = f.add_axes([0.07, 0.27, 0.27, 0.54])
bot = np.zeros(2)
smalls = []
SUMMARY["position_overall"] = {}
for key, lab, cc in ORDER7:
    vals = np.array([(gp.cons == key).mean() * 100, (lp.cons == key).mean() * 100])
    SUMMARY["position_overall"][key] = dict(zip(["gpt-4", "gpt-6-luna"], vals.round(1).tolist()))
    ax.bar([0, 1], vals, 0.55, bottom=bot, color=cc, label=lab)
    for xi, b_, v_ in zip([0, 1], bot, vals):
        if v_ >= 4:
            ax.text(xi, b_ + v_ / 2, f"{v_:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if key in ("consistent", "error") else "#222")
        elif v_ > 0:
            smalls.append((xi, b_ + v_ / 2, v_, cc))
    bot += vals
placed = []
for xi, yc, v_, cc in sorted(smalls, key=lambda t: t[1]):
    y_ = yc if not placed else max(yc, placed[-1] + 4.2)
    placed.append(y_)
    ax.annotate(f"{v_:.1f}%", xy=(xi + 0.28, yc), xytext=(xi + 0.42, y_), fontsize=12, fontweight="bold",
                va="center", arrowprops=dict(arrowstyle="-", color=cc, lw=2))
ax.set_xlim(-0.5, 1.85)
ax.set_xticks([0, 1])
ax.set_xticklabels(["GPT-4\njudge", "gpt-6-luna\njudge"])
ax.set_ylim(0, 100)
ax.set_ylabel("% lượt chấm (n = 800 mỗi judge)")
ax.legend(frameon=False, fontsize=11.5, loc="center left", bbox_to_anchor=(1.03, 0.45))
ax2 = f.add_axes([0.66, 0.27, 0.31, 0.54])
xx = np.arange(8)
cat_cons = {}
for i, (jn, _, d, col) in enumerate(JUDGE_DATA):
    v = d[d.cons != "error"]
    p = v.groupby("cat").cons.apply(lambda s: (s == "consistent").mean() * 100).reindex(CAT_ORDER)
    cat_cons[jn] = p.round(1).to_dict()
    ax2.bar(xx + (i - .5) * 0.38, p, 0.38, color=col, label=f"{jn} judge")
SUMMARY["position_by_category"] = cat_cons
ax2.set_xticks(xx)
ax2.set_xticklabels([CAT_LABEL[c] for c in CAT_ORDER], rotation=40, ha="right", fontsize=12)
ax2.set_ylim(0, 120)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
ax2.set_ylabel("% nhất quán (loại dòng lỗi)")
ax2.set_title("Nhất quán theo category", fontsize=15)
ax2.legend(frameon=False, fontsize=11.5, loc="upper left", ncol=2)
foot(f, "*_pair.jsonl, g1_winner so với g2_winner (g2 = thứ tự đảo). Phân loại theo Bảng 2 của paper. Paper đo GPT-4 "
        "nhất quán 65% trên cặp câu trả lời GẦN GIỐNG NHAU (khó hơn); ở đây là cặp model thật vs gpt-3.5 nên không so trực tiếp.")
save(f, "07_pairwise_consistency.png")

# ---------------------------------------------------------------- 08 agreement
k_ = ["question_id", "model_1", "turn"]
mg = gp[k_ + ["final", "cat"]].merge(lp[k_ + ["final"]], on=k_, suffixes=("_g", "_l"))
v = mg[(mg.final_g != "error") & (mg.final_l != "error")]
nt = v[(v.final_g != "tie") & (v.final_l != "tie")]
SUMMARY["judge_agreement"] = {"S1": round(100 * (v.final_g == v.final_l).mean(), 1), "S1_n": len(v),
                              "S2": round(100 * (nt.final_g == nt.final_l).mean(), 1), "S2_n": len(nt)}
f = fig(f"Đồng thuận GPT-4 – Luna: {SUMMARY['judge_agreement']['S2']:.0f}% khi cả hai chọn thắng, "
        f"{SUMMARY['judge_agreement']['S1']:.0f}% khi tính cả hòa",
        f"Đồng thuận trên kết quả pairwise cuối cùng ({len(v)} lượt chấm hợp lệ ở cả hai judge)")
ax = f.add_axes([0.16, 0.15, 0.31, 0.62])
lab3 = ["win", "tie", "loss"]
ct = pd.crosstab(v.final_g, v.final_l).reindex(index=lab3, columns=lab3).fillna(0)
pct = ct.div(ct.sum(axis=1), axis=0) * 100
ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=75)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{int(ct.values[i, j])}\n({pct.values[i, j]:.0f}%)", ha="center", va="center", fontsize=14,
                fontweight="bold", color="white" if pct.values[i, j] > 45 else "#222")
nm3 = ["Model\nthắng", "Hòa /\nbất nhất", "gpt-3.5\nthắng"]
ax.set_xticks(range(3))
ax.set_xticklabels(nm3, fontsize=11.5)
ax.set_yticks(range(3))
ax.set_yticklabels(nm3, fontsize=11.5)
ax.set_xlabel("gpt-6-luna judge", color=CL)
ax.set_ylabel("GPT-4 judge", color=CG)
ax.set_title("Ma trận nhầm lẫn (% theo hàng)", fontsize=14)
ax.spines[:].set_visible(False)
ax2 = f.add_axes([0.60, 0.15, 0.37, 0.62])
s2 = []
for c in CAT_ORDER:
    y = v[(v.cat == c) & (v.final_g != "tie") & (v.final_l != "tie")]
    s2.append((100 * (y.final_g == y.final_l).mean(), len(y)))
yy = np.arange(8)[::-1]
ax2.barh(yy, [a for a, _ in s2], color=[CG if c in OBJECTIVE else CL for c in CAT_ORDER], height=0.62)
for yi, (a, n_) in zip(yy, s2):
    ax2.text(a + 1, yi, f"{a:.0f}%  (n={n_})", va="center", fontsize=12)
ax2.set_yticks(yy)
ax2.set_yticklabels([CAT_LABEL[c] for c in CAT_ORDER], fontsize=14)
ax2.set_xlim(0, 115)
ax2.set_xlabel("% đồng ý khi cả hai judge chọn người thắng (S2)")
ax2.axvline(SUMMARY["judge_agreement"]["S2"], color="#444", ls="--", lw=1.4)
ax2.legend(handles=[Patch(color=CG, label="Có đáp án / kiểm chứng"), Patch(color=CL, label="Câu hỏi mở")],
           frameon=False, fontsize=11.5, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
foot(f, "*_pair.jsonl ghép theo (question_id, model_1, turn). S1 = tính cả hòa và bất nhất; S2 = chỉ lượt cả hai chọn "
        "người thắng; định nghĩa theo Bảng 5 của paper. Đây là đồng thuận GIỮA HAI JUDGE, không phải với con người.")
save(f, "08_judge_agreement_pairwise.png")

# ---------------------------------------------------------------- 09 ranking bump
rng = np.random.default_rng(5)


def adj_rate(d):
    t = outcomes(d)
    return (t.win + .5 * t.tie) / t.sum(axis=1)


def vg_diff(d):
    vv = d[d["final"] != "error"]
    a = vv[vv.model_1 == "vicuna-13b-v1.3"].groupby("question_id").s.mean()
    b = vv[vv.model_1 == "guanaco-33b"].groupby("question_id").s.mean()
    idx = a.index.intersection(b.index)
    dd = (a[idx] - b[idx]).values
    m, lo, hi = bootstrap_mean(dd, n=5000, seed=1)
    return m, lo, hi, stats.wilcoxon(dd[dd != 0]).pvalue


vg = {jn: vg_diff(d) for jn, _, d, _ in JUDGE_DATA}
SUMMARY["vicuna_minus_guanaco_pair"] = {k: [round(x, 3) for x in v_] for k, v_ in vg.items()}
meas = [("GPT-4\nsingle", gs5.groupby("model").score.mean()), ("GPT-4\npairwise", adj_rate(gp)),
        ("Luna\nsingle", lsv.groupby("model").score.mean()), ("Luna\npairwise", adj_rate(lp))]
f = fig("Thứ hạng ổn định, trừ vicuna–guanaco đổi chỗ",
        "Thứ hạng 5 model theo từng cách chấm (1 = tốt nhất); pairwise xếp theo win-rate vs gpt-3.5-turbo")
ax = f.add_axes([0.08, 0.17, 0.70, 0.64])
cols = {"gpt-4": "#2E5BBA", "claude-v1": "#7A5CC0", "vicuna-13b-v1.3": "#2A9D8F", "guanaco-33b": "#E9A23B",
        "alpaca-13b": "#888"}
R = pd.DataFrame({n: s.loc[MODELS].rank(ascending=False) for n, s in meas})
for mo in MODELS:
    y = R.loc[mo].values
    hl = mo in ("vicuna-13b-v1.3", "guanaco-33b")
    ax.plot(range(4), y, "-o", color=cols[mo], lw=5 if hl else 3, ms=16 if hl else 12, alpha=1 if hl else .6)
    ax.text(3.12, y[-1], SHORT[mo], va="center", fontsize=14, fontweight="bold" if hl else "normal", color=cols[mo])
    for i, (n, s) in enumerate(meas):
        ax.text(i, y[i] - 0.2, f"{s[mo]:.2f}", ha="center", fontsize=11, color="#333")
ax.set_ylim(5.6, 0.4)
ax.set_yticks(range(1, 6))
ax.set_xticks(range(4))
ax.set_xticklabels([n for n, _ in meas])
ax.set_xlim(-.3, 3.9)
ax.set_ylabel("Thứ hạng")
ax.spines[:].set_visible(False)
g_, l_ = vg["gpt-4"], vg["gpt-6-luna"]
f.text(0.80, 0.50, "vicuna − guanaco\n(điểm pairwise theo câu):\n"
                   f"• GPT-4: {g_[0]:+.2f}\n  CI [{g_[1]:.2f}; {g_[2]:.2f}], p = {g_[3]:.2f}\n"
                   f"• Luna: {l_[0]:+.2f}\n  CI [{l_[1]:.3f}; {l_[2]:.2f}], p = {l_[3]:.2f}\n\n"
                   "Không có ý nghĩa ở mức 5%.", fontsize=12, va="center",
       bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#ccc"))
foot(f, "Điểm single: cột score của *_single.jsonl; pairwise: win-rate từ *_pair.jsonl. CI: bootstrap theo câu hỏi.")
save(f, "09_ranking_bump.png")

# ---------------------------------------------------------------- 10 luna failures
bad_single = ls[~ls.valid]
lp_err = lp[lp.cons == "error"]
SUMMARY["luna_errors"] = {"pair": int(len(lp_err)), "single": int(len(bad_single)),
                          "pair_by_prompt": lp_err.prompt_type.value_counts().to_dict(),
                          "gpt4_pair": int((gp.cons == "error").sum()), "gpt4_single": int((~gs5.valid).sum())}
f = fig(f"Độ tin cậy: Luna có {len(lp_err) + len(bad_single)} lượt chấm lỗi, GPT-4 judge có 0",
        "Số lượt chấm lỗi (judgment rỗng) theo loại prompt; tập trung ở prompt có đáp án tham chiếu")
ax = f.add_axes([0.29, 0.17, 0.43, 0.64])
labs = ["pair – reasoning/math/coding\n(nhiều turn)", "pair – reasoning/math/coding\n(turn 1)",
        "pair – câu hỏi mở\n(nhiều turn)", "pair – câu hỏi mở\n(turn 1)", "single – mọi loại"]
vals = [(lp_err.prompt_type == "pair-math-v1-multi-turn").sum(), (lp_err.prompt_type == "pair-math-v1").sum(),
        (lp_err.prompt_type == "pair-v2-multi-turn").sum(), (lp_err.prompt_type == "pair-v2").sum(), len(bad_single)]
yy = np.arange(5)[::-1]
ax.barh(yy, vals, color=CL, height=0.6)
for yi, v_ in zip(yy, vals):
    ax.text(v_ + 0.3, yi, str(v_), va="center", fontsize=15, fontweight="bold")
ax.set_yticks(yy)
ax.set_yticklabels(labs, fontsize=12.5)
ax.set_xlabel("Số lượt chấm lỗi (gpt-6-luna)")
ax.set_xlim(0, max(vals) + 3)
f.text(0.755, 0.50, f"• Pairwise: {len(lp_err)}/800 lượt ({len(lp_err) / 8:.1f}%)\n"
                    f"• Single: {len(bad_single)}/800 dòng ({len(bad_single) / 8:.2f}%)\n"
                    "• GPT-4 judge: 0 lỗi\n\nCode gọi Luna: reasoning_effort='none',\n"
                    "max_completion_tokens=2048,\nkhông đặt temperature.\nNguyên nhân lỗi: chưa xác định.",
       fontsize=12.5, va="center", bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#ccc"))
foot(f, "gpt-6-luna_pair.jsonl (g1_winner/g2_winner = 'error') và gpt-6-luna_single.jsonl (score = -1); "
        "common.py, hàm chat_completion_openai.")
save(f, "10_luna_failures.png")

# ---------------------------------------------------------------- 11 position by model pair (Table 11 style)
pair_rows = []
T11_MODELS = MODELS + ["llama-13b"]
for jn, d in (("gpt-4", gp_all), ("gpt-6-luna", lp_all)):
    for mo in T11_MODELS:
        s = d[d.model_1 == mo]
        if len(s) == 0:
            continue
        s = s[s.cons != "error"]
        row = {"judge": jn, "pair": f"{SHORT[mo]} vs gpt-3.5", "n": len(s)}
        for key in ("consistent", "bias_first", "bias_second", "split_tie"):
            row[key] = round(100 * (s.cons == key).mean(), 1)
        pair_rows.append(row)
pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(os.path.join(OUT_RES, "position_by_pair.csv"), index=False)
f = fig("Thiên kiến vị trí theo cặp model: Luna nhất quán hơn ở mọi cặp",
        "% nhất quán khi đổi thứ tự A/B, từng cặp model vs gpt-3.5-turbo (kiểu Bảng 11 của paper)")
ax = f.add_axes([0.08, 0.17, 0.62, 0.64])
X11 = np.arange(len(T11_MODELS))
for i, (jn, _, _, col) in enumerate(JUDGE_DATA):
    sub = pair_df[pair_df.judge == jn].set_index("pair")
    for xi, mo in zip(X11 + (i - .5) * 0.38, T11_MODELS):
        key = f"{SHORT[mo]} vs gpt-3.5"
        if key in sub.index:
            v_ = sub.loc[key, "consistent"]
            ax.bar(xi, v_, 0.38, color=col, label=f"{jn} judge" if mo == MODELS[0] else None)
            ax.text(xi, v_ + 1.5, f"{v_:.0f}", ha="center", fontsize=12, fontweight="bold")
        else:
            ax.text(xi, 3, "chưa\nchạy", ha="center", fontsize=10, color=col)
ax.set_xticks(X11)
ax.set_xticklabels([f"{SHORT[m]}\nvs gpt-3.5" for m in T11_MODELS], fontsize=11)
ax.set_ylim(0, 118)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel("% nhất quán")
ax.legend(frameon=False, loc="upper left", ncol=2)
f.text(0.74, 0.50, "Paper (GPT-4, Bảng 11):\n• gpt-3.5 vs claude-v1: 67.5%\n• gpt-3.5 vs vicuna-13b: 73.8%\n"
                   "• gpt-3.5 vs llama-13b: 98.8%\n\nPaper thấy thiên kiến giảm khi\nhai model chênh lệch lớn.\n"
                   "Cặp llama-13b của Luna: chạy\ngen_judgment.py (xem README).", fontsize=12.5, va="center",
       bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#ccc"))
foot(f, "*_pair.jsonl, g1_winner vs g2_winner theo model_1; loại dòng lỗi. Số của paper dùng prompt và cách lấy mẫu "
        "riêng nên chỉ để tham khảo xu hướng.")
save(f, "11_position_by_pair.png")


# ---------------------------------------------------------------- 12 verbosity on real answers
def add_lengths(d):
    d = d[d["final"] != "error"].copy()
    d["len1"] = [words(ans[(m, q, t)]) for m, q, t in zip(d.model_1, d.question_id, d.turn)]
    d["len2"] = [words(ans[(BASELINE, q, t)]) for q, t in zip(d.question_id, d.turn)]
    d["z"] = (d.len1 - d.len2) / (d.len1 + d.len2).clip(lower=1)
    return d


def fit_gamma(d):
    """Logistic regression, LMSYS style-control form:
       logit P(model_1 wins) = beta_model_1 + gamma * z_len.
       beta absorbs model quality (one coefficient per model, baseline fixed).
       Ties = half win + half loss."""
    mods = MODELS
    Xm = np.column_stack([(d.model_1 == m).astype(float) for m in mods] + [d.z.values])
    y = d.s.values
    Xw = np.vstack([Xm, Xm])
    yw = np.concatenate([np.ones(len(y)), np.zeros(len(y))])
    ww = np.concatenate([y, 1 - y])
    beta = np.zeros(Xw.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-Xw @ beta))
        g = Xw.T @ (ww * (yw - p)) - 1e-4 * beta
        H = -(Xw * (ww * p * (1 - p))[:, None]).T @ Xw - 1e-4 * np.eye(len(beta))
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta[-1]


f = fig("Độ dài trên câu trả lời thật: GPT-4 ưa câu dài, Luna thì không",
        "Giữ cố định chất lượng model, câu trả lời dài hơn có làm tăng khả năng thắng không? (pairwise vs gpt-3.5)")
ax = f.add_axes([0.07, 0.22, 0.40, 0.58])
gam = {}
for i, (jn, _, d, col) in enumerate(JUDGE_DATA):
    dl = add_lengths(d)
    m, lo, hi = bootstrap_by_question(dl, fit_gamma, n=1000, seed=3)
    gam[jn] = (m, lo, hi)
    ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=16, color=col, capsize=8, lw=2.5)
    ax.text(i + 0.12, m, f"γ = {m:.2f}\n[{lo:.2f}; {hi:.2f}]".replace("-", "−"), va="center", fontsize=13)
SUMMARY["verbosity_gamma"] = {k: [round(x, 3) for x in v_] for k, v_ in gam.items()}
ax.axhline(0, color="#555", lw=1.2)
ax.set_xticks([0, 1])
ax.set_xticklabels(["GPT-4 judge", "gpt-6-luna judge"])
ax.set_xlim(-0.5, 1.9)
ax.set_ylabel("γ độ dài (log-odds)")
ax.set_title("Hệ số ưa độ dài (95% CI)", fontsize=15)
# descriptive panel: win prob of longer answer by length-ratio bins, both judges
ax2 = f.add_axes([0.57, 0.22, 0.40, 0.58])
bins = [-1, -0.33, -0.1, 0.1, 0.33, 1]
blab = ["ngắn hơn\nnhiều", "ngắn\nhơn", "tương\nđương", "dài\nhơn", "dài hơn\nnhiều"]
for i, (jn, _, d, col) in enumerate(JUDGE_DATA):
    dl = add_lengths(d)
    dl = dl[dl.model_1 != "alpaca-13b"]  # alpaca is both short and weak; exclude to reduce confounding
    dl["bin"] = pd.cut(dl.z, bins, labels=blab)
    p = dl.groupby("bin", observed=False).s.mean()
    ax2.plot(range(5), p.values, "-o", color=col, lw=3, ms=10, label=f"{jn} judge")
ax2.axhline(0.5, color="#aaa", ls="--")
ax2.set_xticks(range(5))
ax2.set_xticklabels(blab, fontsize=11)
ax2.set_ylim(0, 1)
ax2.set_ylabel("Điểm pairwise TB của model\n(thắng = 1, hòa = 0.5)")
ax2.set_xlabel("Độ dài so với câu của gpt-3.5")
ax2.set_title("Mô tả (bỏ alpaca)", fontsize=15)
ax2.legend(frameon=False, loc="upper left", fontsize=12)
foot(f, "*_pair.jsonl + model_answer/*.jsonl (độ dài = số từ của câu trả lời ở turn được chấm). γ: hồi quy logistic có hệ "
        "số riêng cho từng model (kiểu style control của LMSYS), hòa = nửa thắng nửa thua, bootstrap theo câu hỏi 1.000 lần. "
        "γ > 0 chỉ cho biết judge ưa câu dài; muốn biết có VƯỢT MỨC NGƯỜI không cần phiếu chuyên gia (Δγ, phương án V2).")
save(f, "12_verbosity_real_data.png")

# single-answer partial correlation: score vs log length, removing model and (question, turn) means
part = {}
for jn, d, _, _ in JUDGE_DATA:
    d = d.copy()
    d["loglen"] = np.log1p([words(ans[(m, q, t)]) for m, q, t in zip(d.model, d.question_id, d.turn)])

    def pcorr(x):
        r = {}
        for c in ("score", "loglen"):
            v_ = x[c] - x.groupby("model")[c].transform("mean")
            v_ = v_ - v_.groupby([x.question_id, x.turn]).transform("mean")
            r[c] = v_
        return stats.pearsonr(r["score"], r["loglen"])[0]

    part[jn] = [round(v_, 3) for v_ in bootstrap_by_question(d, pcorr, n=1000, seed=4)]
SUMMARY["single_partial_corr_score_length"] = part

with open(os.path.join(OUT_RES, "summary.json"), "w") as fo:
    json.dump(SUMMARY, fo, indent=2, ensure_ascii=False)
print(json.dumps(SUMMARY, indent=1, ensure_ascii=False))
