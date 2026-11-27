import json
import pandas as pd


def load_judge_map(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("results", data.get("items", []))
    return {str(x["id"]): x for x in items}


def decode_winner(raw_val, first_model: str, second_model: str) -> str:
    val = str(raw_val).strip().upper()
    if val in ("C1", "A", "ASSISTANT_A", "CANDIDATE_1", "1"):
        return first_model
    if val in ("C2", "B", "ASSISTANT_B", "CANDIDATE_2", "2"):
        return second_model
    if val in ("TIE", "平", "平局", "DRAW", "SAME"):
        return "tie"
    raise ValueError(f"无法识别的胜负标记: {raw_val!r}（期望 C1/C2/Tie 或 A/B/Tie）")


# 1. 加载人工盲评表、本地映射与正反序两轮 AI 裁判结果
human_df = pd.read_excel("blind_sheet_for_human.xlsx")
if human_df[["human_c1_hard_pass", "human_c2_hard_pass", "human_winner(C1/C2/Tie)"]].isna().any().any():
    raise SystemExit("错误：blind_sheet_for_human.xlsx 存在未填写的盲评单元格，请先填写完整。")
with open("secret_mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)
r1_map = load_judge_map("judge_result_round1.json")
r2_map = load_judge_map("judge_result_round2.json")

# 2. 逐题还原真实模型身份并汇总明细
records = []
for _, row in human_df.iterrows():
    qid = str(row["id"])
    m, r1, r2 = mapping[qid], r1_map[qid], r2_map[qid]
    c1_pass = int(float(str(row["human_c1_hard_pass"]).strip()))
    c2_pass = int(float(str(row["human_c2_hard_pass"]).strip()))
    h_db_pass = c1_pass if m["Candidate_1"] == "doubao" else c2_pass
    h_ds_pass = c1_pass if m["Candidate_1"] == "deepseek" else c2_pass
    h_win = decode_winner(row["human_winner(C1/C2/Tie)"], m["Candidate_1"], m["Candidate_2"])
    r1_win = decode_winner(r1["winner"], "doubao", "deepseek")
    r2_win = decode_winner(r2["winner"], "deepseek", "doubao")

    records.append({
        "id": qid, "category": row["category"], "prompt": row["prompt"], "rubric": row["rubric"],
        "doubao_len": m["doubao_len"], "deepseek_len": m["deepseek_len"],
        "human_doubao_pass": h_db_pass, "human_deepseek_pass": h_ds_pass, "human_winner": h_win,
        "human_badcase_note": "" if pd.isna(row["human_badcase_note"]) else str(row["human_badcase_note"]).strip(),
        "r1_doubao_pass": int(r1["a_hard_pass"]), "r1_deepseek_pass": int(r1["b_hard_pass"]), "r1_winner": r1_win,
        "r2_doubao_pass": int(r2["b_hard_pass"]), "r2_deepseek_pass": int(r2["a_hard_pass"]), "r2_winner": r2_win,
        "judge_agree_human": int(r1_win == h_win), "position_flipped": int(r1_win != r2_win),
        "r1_reason": r1.get("reason", ""), "r2_reason": r2.get("reason", ""),
        "doubao_answer": m["doubao_answer"], "deepseek_answer": m["deepseek_answer"],
    })

df = pd.DataFrame(records)
df.to_csv("full_eval_results.csv", index=False, encoding="utf-8-sig")

# 3. 计算分维度硬约束通过率与人工 GSB 胜平负
cat_lines = []
for cat, g in list(df.groupby("category", sort=False)) + [("总计 (Overall)", df)]:
    db_p, ds_p = g["human_doubao_pass"].mean() * 100, g["human_deepseek_pass"].mean() * 100
    r1_db, r1_ds = g["r1_doubao_pass"].mean() * 100, g["r1_deepseek_pass"].mean() * 100
    gsb = f"{int((g['human_winner']=='doubao').sum())} / {int((g['human_winner']=='tie').sum())} / {int((g['human_winner']=='deepseek').sum())}"
    cat_lines.append(f"| {cat} | {len(g)} | {db_p:.1f}% (AI:{r1_db:.1f}%) | {ds_p:.1f}% (AI:{r1_ds:.1f}%) | {gsb} |")

# 4. 计算人机一致率、正反序翻盘率（Position Bias）及字数偏好（Verbosity Bias）
total = len(df)
win_agree = df["judge_agree_human"].mean() * 100
pass_agree = ((df["r1_doubao_pass"] == df["human_doubao_pass"]).sum()
              + (df["r1_deepseek_pass"] == df["human_deepseek_pass"]).sum()) / (2 * total) * 100
flip_rate = df["position_flipped"].mean() * 100
decided = df[df["r1_winner"].isin(["doubao", "deepseek"])]
win_lens = [r["doubao_len"] if r["r1_winner"] == "doubao" else r["deepseek_len"] for _, r in decided.iterrows()]
lose_lens = [r["deepseek_len"] if r["r1_winner"] == "doubao" else r["doubao_len"] for _, r in decided.iterrows()]
avg_w = sum(win_lens) / len(win_lens) if win_lens else 0.0
avg_l = sum(lose_lens) / len(lose_lens) if lose_lens else 0.0
long_win = (sum(w > l for w, l in zip(win_lens, lose_lens)) / len(win_lens) * 100) if win_lens else 0.0

# 5. 输出 Markdown 评测汇总报告
report = [
    "# 豆包 vs DeepSeek 盲评汇总报告\n",
    "## 1. 分维度硬约束通过率与人工 GSB 胜平负",
    "| 评测维度 | 题数 | 豆包硬约束通过率 | DeepSeek硬约束通过率 | 人工 GSB (豆包胜 / 平 / DeepSeek胜) |",
    "| :--- | :---: | :---: | :---: | :---: |",
    *cat_lines,
    "\n## 2. AI 裁判（正序 Round1）与人工盲评一致率 (Agreement Rate)",
    f"- **GSB 胜负判定一致率**: {int(df['judge_agree_human'].sum())}/{total} (**{win_agree:.1f}%**)",
    f"- **0/1 硬约束核查一致率**: **{pass_agree:.1f}%**（共 {2 * total} 次单模型判定）\n",
    "## 3. AI 裁判偏见分析（Position Bias & Verbosity Bias）",
    f"- **正反序对调翻盘率 (Position Bias Rate)**: {int(df['position_flipped'].sum())}/{total} (**{flip_rate:.1f}%**)",
    f"- **模型整体平均字数**: 豆包 **{df['doubao_len'].mean():.1f}** 字 vs DeepSeek **{df['deepseek_len'].mean():.1f}** 字",
    f"- **AI 裁判非平局场次字数对比 (Verbosity Bias)**: 胜出方平均 **{avg_w:.1f}** 字 vs 落败方平均 **{avg_l:.1f}** 字（更长回答胜出占比 **{long_win:.1f}%**）",
]
with open("eval_summary_metrics.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")

print("统计完毕！已生成 eval_summary_metrics.md 与 full_eval_results.csv。")
