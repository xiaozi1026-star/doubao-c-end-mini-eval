import json
import os
import random
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载并校验 .env 配置
load_dotenv(override=True)
REQUIRED_KEYS = [
    "ARK_API_KEY", "ARK_BASE_URL", "ARK_MODEL",
    "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
]
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    raise SystemExit(f"错误：.env 缺少必要配置项：{', '.join(missing)}")

ark_client = OpenAI(api_key=os.getenv("ARK_API_KEY"), base_url=os.getenv("ARK_BASE_URL"), timeout=60.0)
ds_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL"), timeout=60.0)
ark_model = os.getenv("ARK_MODEL")
ds_model = os.getenv("DEEPSEEK_MODEL")


def ask_model(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (resp.choices[0].message.content or "").strip()


# 2. 读取题库并逐题调用豆包与 DeepSeek（强制关闭深度思考）
questions_df = pd.read_csv("questions.csv", encoding="utf-8-sig")
rng = random.Random()
human_rows, round1_items, round2_items = [], [], []
secret_mapping = {}

for idx, row in questions_df.iterrows():
    qid = str(row["id"])
    print(f"[{idx + 1}/{len(questions_df)}] 正在请求回答: {qid} ({row['category']}) ...", flush=True)
    doubao_ans = ask_model(ark_client, ark_model, row["prompt"])
    deepseek_ans = ask_model(ds_client, ds_model, row["prompt"])

    # 3. 随机打乱人工盲评顺序 Candidate_1 / Candidate_2
    if rng.choice([True, False]):
        c1_name, c2_name = "doubao", "deepseek"
        c1_text, c2_text = doubao_ans, deepseek_ans
    else:
        c1_name, c2_name = "deepseek", "doubao"
        c1_text, c2_text = deepseek_ans, doubao_ans

    human_rows.append({
        "id": qid,
        "category": row["category"],
        "prompt": row["prompt"],
        "rubric": row["rubric"],
        "Candidate_1": c1_text,
        "Candidate_2": c2_text,
        "human_c1_hard_pass": "",
        "human_c2_hard_pass": "",
        "human_winner(C1/C2/Tie)": "",
        "human_badcase_note": "",
    })
    secret_mapping[qid] = {
        "Candidate_1": c1_name,
        "Candidate_2": c2_name,
        "doubao_len": len(doubao_ans),
        "deepseek_len": len(deepseek_ans),
        "doubao_answer": doubao_ans,
        "deepseek_answer": deepseek_ans,
    }

    # 4. 构造双轮 AI 盲评数据包（不暴露模型真名，Round1 正序 / Round2 对调）
    base = {"id": qid, "category": row["category"], "prompt": row["prompt"], "rubric": row["rubric"]}
    round1_items.append({**base, "Assistant_A": doubao_ans, "Assistant_B": deepseek_ans})
    round2_items.append({**base, "Assistant_A": deepseek_ans, "Assistant_B": doubao_ans})

# 5. 导出人工盲评表、本地真实映射与双轮 AI 裁判盲评包
pd.DataFrame(human_rows).to_excel("blind_sheet_for_human.xlsx", index=False)
with open("secret_mapping.json", "w", encoding="utf-8") as f:
    json.dump(secret_mapping, f, ensure_ascii=False, indent=2)

judge_instruction = (
    "请作为独立盲评裁判逐题核查 Assistant_A 与 Assistant_B，严禁猜测模型身份。"
    "输出 JSON 数组，每项含: id, a_hard_pass(0/1), b_hard_pass(0/1), winner('A'/'B'/'Tie'), reason。"
)
with open("judge_pack_round1.json", "w", encoding="utf-8") as f:
    json.dump({"instruction": judge_instruction, "items": round1_items}, f, ensure_ascii=False, indent=2)
with open("judge_pack_round2.json", "w", encoding="utf-8") as f:
    json.dump({"instruction": judge_instruction, "items": round2_items}, f, ensure_ascii=False, indent=2)

print("生成完毕！已输出 blind_sheet_for_human.xlsx、secret_mapping.json 及两份 judge_pack。")
