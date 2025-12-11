import os
import json
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report

from text_encoder import flow_to_text


# 为了控制费用 & 速度，可以先只抽一部分样本做 baseline
MAX_SAMPLES = 100          # 你可以改成 50 / 100 / 500 ...
SLEEP_BETWEEN_CALLS = 0.1  # 每次调用之间稍微停一下，避免过快触发限速


def build_prompt(flow_text: str) -> str:
    return f"""
You are a senior APT threat analyst working in a cyber threat intelligence (CTI) team.
Your mission is to aggressively hunt for signs of advanced persistent threats (APT) in enterprise network traffic.

APT attacks are stealthy and often hide inside seemingly normal traffic. Therefore:
- If you see any network behavior that is unusual, rare, suspicious, or atypical, you should strongly consider classifying it as APT.
- False negatives (missing an APT) are considered FAR more dangerous than false positives in this task.
- You are encouraged to be conservative toward safety: when uncertain, lean toward labeling "APT".

Common APT indicators include (but are not limited to):
- Rare or abnormal destination domains
- Unusual port usage
- Encrypted traffic with abnormal patterns
- Anomalous packet timing patterns
- Strange application behaviors
- Indicators of command & control communication
- Laser-focused low-volume communication
- Traffic that does not match typical user behavior
- Suspicious or outdated User-Agent strings
- JA3 / JA4 fingerprints associated with malware families (even if you are unsure)

Now analyze the following network flow like a real APT hunter:

Flow description:
\"\"\"
{flow_text}
\"\"\"

Carefully reason as an expert analyst.

Return your answer in STRICT JSON format ONLY:

{{
  "label": "APT" or "Benign",
  "confidence": "high / medium / low",
  "reason": "short explanation focusing on threat indicators"
}}

DO NOT output anything else.
""".strip()



def parse_label_from_response(text: str) -> str | None:
    """
    从模型返回的文本中解析出 label。
    期望是一个 JSON 字符串，如果解析失败就返回 None。
    """
    text = text.strip()
    # 有些模型会在 JSON 前后加废话，我们简单截取第一个 '{' 到最后一个 '}' 之间
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]

    try:
        data = json.loads(text)
        label = str(data.get("label", "")).strip()
        # 统一成大写首字母，防止出现 apt / benign 之类
        label_upper = label.upper()
        if label_upper == "APT":
            return "APT"
        if label_upper == "BENIGN":
            return "Benign"
        # 如果其他奇怪输出，直接返回 None，由外面处理
        return None
    except Exception:
        return None


def main():
    # 1. 读取 .env，初始化 OpenAI 客户端
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未配置，请先在 .env 中设置。")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 2. 读取测试集
    df_test = pd.read_csv("../data/test.csv", low_memory=False)
    print(f"Loaded test dataset, total rows: {len(df_test)}")

    # 随机抽样，控制调用成本
    n_samples = min(MAX_SAMPLES, len(df_test))
    df_sample = df_test.sample(n=n_samples, random_state=42)
    print(f"Sampling {n_samples} rows for LLM baseline evaluation.")
    print("-" * 80)

    y_true: list[str] = []
    y_pred: list[str] = []

    for i, (idx, row) in enumerate(df_sample.iterrows(), start=1):
        true_label = row["label_sub"]

        # 3. 编码为文本
        flow_text = flow_to_text(row)

        # 4. 构造 prompt
        prompt = build_prompt(flow_text)

        # 5. 调用大模型
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # baseline 先设为 0，保证稳定
            )
            content = resp.choices[0].message.content
        except Exception as e:
            print(f"[{i}/{n_samples}] Error calling LLM on index {idx}: {e}")
            # 调用失败时可以选择跳过或给一个默认预测
            pred_label = "Benign"
        else:
            # 6. 解析预测标签
            pred_label = parse_label_from_response(content)
            if pred_label is None:
                print(f"[{i}/{n_samples}] Failed to parse label for index {idx}, raw response:")
                print(content)
                # 简单兜底：预测失败时，可以默认判为 Benign（或者直接跳过）
                pred_label = "Benign"

        y_true.append(true_label)
        y_pred.append(pred_label)

        # 打个进度
        if i % 10 == 0 or i == n_samples:
            print(f"[Progress] {i}/{n_samples} samples processed.")

        # 防止请求太密集
        time.sleep(SLEEP_BETWEEN_CALLS)

    print("\n=== LLM Baseline Evaluation on Sampled Test Set ===")
    # 固定标签顺序，方便阅读
    labels = ["APT", "Benign"]
    print(classification_report(y_true, y_pred, labels=labels))


if __name__ == "__main__":
    main()
