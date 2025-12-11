import os
import json
import time
from typing import Tuple, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report

from text_encoder import flow_to_text


# 为了控制费用 & 速度，可以先只抽一部分样本做 baseline
MAX_SAMPLES = 100          # 你可以改成 50 / 100 / 500 ...
SLEEP_BETWEEN_CALLS = 0.1  # 每次调用之间稍微停一下，避免过快触发限速


# ====== 1. 策略配置：现在包含 Prompt + Gate 多个维度 ======
policy_config: Dict[str, Any] = {
    # ---- Prompt 相关 ----
    # 整体风格：影响“安全偏置”的强弱
    # - "aggressive": 强烈偏向 APT（安全优先）
    # - "balanced" : 居中
    # - "conservative": 谨慎，只在证据较强时报 APT
    "prompt_variant": "aggressive",

    # APT 特征提示的详细程度：low / medium / high
    "indicator_level": "high",

    # 是否在 Prompt 中明确强调“漏报 APT 比误报更危险”
    "emphasize_apt": True,

    # ---- Gate 相关 ----
    # 门控模式：
    #   "none"      : 不做门控，直接使用 LLM 输出的 label
    #   "threshold" : 使用置信度阈值对 APT 进行降级
    "gate_mode": "threshold",

    # 置信度阈值：
    #   high   : 仅 high 才保留为 APT
    #   medium : medium / high 保留为 APT
    #   low    : low / medium / high 全部允许为 APT（几乎等于不降级）
    "conf_threshold": "high",

    # 也可以把 temperature 作为策略参数（可选）
    "temperature": 0.0,
}


# ====== 2. Prompt 模板库：不同风格的固定文本骨架 ======

PROMPT_VARIANTS: Dict[str, str] = {
    # 强调“宁可多报APT”的版本
    "aggressive": """
You are a senior APT threat analyst working in a cyber threat intelligence (CTI) team.
Your primary mission is to aggressively hunt for signs of advanced persistent threats (APT) in enterprise network traffic.

APT attacks are stealthy and often hide inside seemingly normal traffic. Missing an APT is considered far more dangerous than raising a false alarm.
When there is reasonable doubt, you should lean toward classifying the flow as "APT", as long as there are concrete suspicious indicators.

""".strip(),

    # 兼顾误报和漏报的版本
    "balanced": """
You are a senior APT threat analyst working in a cyber threat intelligence (CTI) team.
Your mission is to carefully assess whether a network flow is more likely to be part of an APT attack or benign traffic.

You should weigh both benign explanations and malicious indicators. You must be cautious about over-reacting to minor anomalies, but you must not ignore strong evidence of APT-like behavior.

""".strip(),

    # 非常谨慎的版本：只在强证据下给 APT
    "conservative": """
You are a senior APT threat analyst working in a cyber threat intelligence (CTI) team.
Your mission is to only label a flow as an APT when there are multiple strong and consistent indicators of targeted, persistent malicious behavior.

Whenever the evidence is weak, ambiguous, or can be easily explained by normal user activity, you should classify the flow as "Benign".

""".strip(),
}


def build_indicator_block(indicator_level: str) -> str:
    """
    根据 indicator_level 决定列出多少 APT 相关线索。
    """
    base = []

    # 所有等级都会包含的基础提示
    base.append("- Destination domain reputation and rarity")
    base.append("- Port usage patterns and protocol consistency")
    base.append("- Timing and volume characteristics of the flow")

    if indicator_level in ("medium", "high"):
        base.append("- Whether behavior matches typical user or application patterns")
        base.append("- Suspicious or outdated User-Agent strings")
        base.append("- Signs of beaconing or low-volume but persistent communication")

    if indicator_level == "high":
        base.append("- JA3 / JA4 TLS fingerprints that may be associated with malware families")
        base.append("- Potential command & control (C2) characteristics")
        base.append("- Unusual encryption or payload size distributions")
        base.append("- Any anomalies that would concern an experienced threat hunter")

    lines = "\n".join(f"{line}" for line in base)
    return (
        "You should pay special attention to the following indicators:\n"
        + lines
        + "\n"
    )


def build_emphasis_block(emphasize_apt: bool, variant: str) -> str:
    """
    用一个小块文字强调 / 弱化 “漏报 APT 更危险” 的思想。
    """
    if not emphasize_apt:
        # 不强调时给一个中性提醒
        return (
            "You must balance the risk of missing an APT with the cost of false alarms. "
            "Always ground your decision in observable technical indicators.\n"
        )

    # 强调时，根据 variant 稍微调整语气（可以以后细化）
    if variant == "aggressive":
        return (
            "In this task, missing a real APT is considered far more dangerous than raising a false alarm. "
            "When there is significant doubt and several suspicious indicators, prefer labeling as \"APT\".\n"
        )
    elif variant == "balanced":
        return (
            "You should slightly favor catching APTs over minimizing false positives, "
            "as long as your decision is supported by concrete technical indicators.\n"
        )
    else:  # conservative + emphasize_apt
        return (
            "Even though you are conservative, you must not miss clear signs of APT. "
            "If multiple strong indicators align, do not hesitate to label as \"APT\".\n"
        )


def build_prompt(flow_text: str, cfg: Dict[str, Any]) -> str:
    """
    根据策略配置构造 Prompt。
    - prompt_variant: aggressive / balanced / conservative
    - indicator_level: low / medium / high
    - emphasize_apt: True / False
    """
    variant = cfg.get("prompt_variant", "aggressive")
    indicator_level = cfg.get("indicator_level", "medium")
    emphasize_apt = cfg.get("emphasize_apt", True)

    role_block = PROMPT_VARIANTS.get(variant, PROMPT_VARIANTS["aggressive"])
    indicator_block = build_indicator_block(indicator_level)
    emphasis_block = build_emphasis_block(emphasize_apt, variant)

    prompt = f"""
{role_block}

{emphasis_block}
{indicator_block}
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

    return prompt


# ====== 3. 解析 LLM 输出 + Gate 逻辑 ======

def parse_response(text: str) -> Tuple[str | None, str | None]:
    """
    从模型返回的文本中解析出 (label, confidence)。
    期望 JSON：
    {
      "label": "APT" or "Benign",
      "confidence": "high / medium / low",
      ...
    }
    """
    text = text.strip()

    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]

    try:
        data = json.loads(text)
    except Exception:
        return None, None

    raw_label = str(data.get("label", "")).strip()
    label_upper = raw_label.upper()
    if label_upper == "APT":
        label_norm = "APT"
    elif label_upper == "BENIGN":
        label_norm = "Benign"
    else:
        label_norm = None

    raw_conf = str(data.get("confidence", "")).strip().lower()
    if raw_conf in ("high", "medium", "low"):
        conf_norm = raw_conf
    else:
        conf_norm = None

    return label_norm, conf_norm


def apply_gate(label_raw: str | None, conf: str | None, cfg: Dict[str, Any]) -> Tuple[str | None, bool]:
    """
    根据策略配置，对原始预测 label + confidence 进行“门控”。

    返回:
      final_label: "APT" / "Benign" / None
      downgraded:  bool，是否发生过从 APT -> Benign 的降级
    """
    gate_mode = cfg.get("gate_mode", "none")
    conf_threshold = cfg.get("conf_threshold", "high")

    if label_raw is None or label_raw != "APT":
        return label_raw, False

    if gate_mode == "none":
        return label_raw, False

    level_map = {"low": 1, "medium": 2, "high": 3}
    conf_level = level_map.get(conf or "", 0)
    threshold_level = level_map.get(conf_threshold, 3)

    if conf_level >= threshold_level:
        return "APT", False
    else:
        return "Benign", True


# ====== 4. 单次 run：给定 policy_config，评估在一个 sample 上的表现 ======

def evaluate_policy_on_dataset(
    df: pd.DataFrame,
    client: OpenAI,
    model_name: str,
    cfg: Dict[str, Any],
    max_samples: int = MAX_SAMPLES,
    verbose: bool = True,
):
    """
    这是后面可以直接给 Agent Lightning 调用的“黑盒目标函数”核心：
    输入: policy_config
    输出: (avg_reward, report)
    这里只先返回 classification_report，reward 你可以再设计。
    """
    n_samples = min(max_samples, len(df))
    df_sample = df.sample(n=n_samples, random_state=42)

    y_true: list[str] = []
    y_pred: list[str] = []
    downgraded_count = 0

    for i, (idx, row) in enumerate(df_sample.iterrows(), start=1):
        true_label = row["label_sub"]
        flow_text = flow_to_text(row)
        prompt = build_prompt(flow_text, cfg)

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.get("temperature", 0.0),
            )
            content = resp.choices[0].message.content
        except Exception as e:
            if verbose:
                print(f"[{i}/{n_samples}] Error calling LLM on index {idx}: {e}")
            pred_label = "Benign"
        else:
            label_raw, conf = parse_response(content)
            if label_raw is None:
                if verbose:
                    print(f"[{i}/{n_samples}] Failed to parse label for index {idx}, raw response:")
                    print(content)
                pred_label = "Benign"
            else:
                pred_label, downgraded = apply_gate(label_raw, conf, cfg)
                if downgraded:
                    downgraded_count += 1

        y_true.append(true_label)
        y_pred.append(pred_label)

        if verbose and (i % 10 == 0 or i == n_samples):
            print(f"[Progress] {i}/{n_samples} samples processed.")

        time.sleep(SLEEP_BETWEEN_CALLS)

    # 这里的 report 既可以打印看，也可以作为后续 reward 计算依据
    report = classification_report(
        y_true,
        y_pred,
        labels=["APT", "Benign"],
        output_dict=True,
        zero_division=0,
    )

    if verbose:
        print("\n=== LLM Evaluation on Sampled Test Set (with policy_config) ===")
        print(f"Policy config: {cfg}")
        print(f"APT predictions downgraded to Benign due to gate: {downgraded_count}")
        # 再打印一个可读版
        print(classification_report(y_true, y_pred, labels=["APT", "Benign"], zero_division=0))

    return report, downgraded_count


def main():
    # 1. 读取 .env，初始化 OpenAI 客户端
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未配置，请先在 .env 中设置。")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 2. 读取测试集
    df_test = pd.read_csv("../data/test.csv", low_memory=False)
    print(f"Loaded test dataset, total rows: {len(df_test)}")

    # 3. 用当前 policy_config 跑一遍（手工模式 / 调参前验证）
    report, downgraded_count = evaluate_policy_on_dataset(
        df=df_test,
        client=client,
        model_name=model,
        cfg=policy_config,
        max_samples=MAX_SAMPLES,
        verbose=True,
    )


if __name__ == "__main__":
    main()
