import os
import json
import random
from typing import Dict, Any, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from sklearn.model_selection import train_test_split

import agentlightning as agl
from agentlightning.instrumentation import instrument_all

from text_encoder import flow_to_text


# ============= 一些全局超参数（可以以后再调） =============
MAX_TASKS = 100          # 为了省钱，先用最多 200 条流做训练/验证
CONF_THRESHOLD = "high"  # gating 使用的置信度阈值（目前先写死成 high）


# ============= 0. 全局 LLM 客户端（同步版） =============

_sync_client: OpenAI | None = None


def get_sync_client() -> OpenAI:
    """
    懒加载一个全局的同步 OpenAI 客户端，避免在每个 rollout 里重复初始化。
    """
    global _sync_client
    if _sync_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置，请先在 .env 中设置。")
        _sync_client = OpenAI(api_key=api_key, base_url=base_url)
    return _sync_client


# ============= 1. 基础 Prompt 模板（会被 APO 自动改写） =============

def prompt_template_baseline() -> agl.PromptTemplate:
    """
    定义一个带占位符 {flow_text} 的 PromptTemplate，作为 APO 的初始资源。
    后续 APO 会对这个 template 的内容做自然语言级别的修改。
    注意：这里故意不再写示例 JSON 里的大括号，避免 str.format 误解析。
    """
    template_str = """
You are a senior APT threat analyst working in a cyber threat intelligence (CTI) team.
Your mission is to analyze a SINGLE network flow and decide whether it is part of an APT attack or benign traffic.

Background:
- APT attacks are stealthy and may hide inside seemingly normal traffic.
- Missing a real APT is more dangerous than raising a false alarm.
- However, you should still ground your decision in concrete, observable indicators.

When making your decision, pay attention to:
- Destination domain reputation and rarity
- Port usage and protocol consistency
- Timing and volume characteristics (beaconing, periodic low-volume C2, etc.)
- Whether the behavior matches typical user or application patterns
- Suspicious or outdated User-Agent strings
- Possible signs of command & control (C2) communication
- TLS fingerprints (JA3 / JA4) that may relate to malware families
- Unusual encryption or payload size distributions

You must carefully read the flow description below and then decide:

Flow description:
\"\"\" 
{flow_text}
\"\"\"

Return your answer in STRICT JSON format ONLY, using exactly these keys:
- "label": "APT" or "Benign"
- "confidence": "high" or "medium" or "low"
- "reason": a VERY short explanation in English, focusing on the key indicators

Requirements:
- Output MUST be valid JSON.
- Do NOT include any extra fields.
- Do NOT include any explanations outside the JSON itself.
""".strip()

    return agl.PromptTemplate(
        template=template_str,
        engine="f-string",
    )



# ============= 2. 解析 LLM 输出 + gating 逻辑 =============

def parse_response(text: str) -> Tuple[str | None, str | None]:
    """
    从模型返回文本中解析 (label_norm, conf_norm)。
    期望格式：
      { "label": "APT" or "Benign", "confidence": "high"/"medium"/"low", ... }
    """
    text = text.strip()
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]

    try:
        data = json.loads(text)
    except Exception:
        return None, None

    raw_label = str(data.get("label", "")).strip().upper()
    if raw_label == "APT":
        label_norm = "APT"
    elif raw_label == "BENIGN":
        label_norm = "Benign"
    else:
        label_norm = None

    raw_conf = str(data.get("confidence", "")).strip().lower()
    if raw_conf in ("low", "medium", "high"):
        conf_norm = raw_conf
    else:
        conf_norm = None

    return label_norm, conf_norm


def apply_gate(label_raw: str | None,
               conf: str | None,
               threshold: str = CONF_THRESHOLD) -> Tuple[str | None, bool]:
    """
    简化版 gating：
    - threshold = "none"      => 不做 gating
    - threshold = "low"       => low/medium/high 都视为足够 => 等于不降级
    - threshold = "medium"    => 需要 medium 或 high
    - threshold = "high"      => 只接受 high 置信度的 APT

    返回:
      (final_label, downgraded_flag)
    """
    # 非 APT 不参与降级逻辑
    if label_raw != "APT":
        return label_raw, False

    if threshold == "none":
        return "APT", False

    level_map = {"low": 1, "medium": 2, "high": 3}
    conf_level = level_map.get(conf or "", 0)
    thr_level = level_map.get(threshold, 3)

    if conf_level >= thr_level:
        return "APT", False
    else:
        return "Benign", True


# ============= 3. 构造 Task 数据集 =============

def build_tasks_from_df(df: pd.DataFrame, max_tasks: int) -> List[Dict[str, Any]]:
    """
    把 DataFrame 里的每一行转换成一个 task:
      { "flow_text": str, "true_label": "APT"/"Benign" }
    """
    # 先随机采样，避免成本太高
    if len(df) > max_tasks:
        df = df.sample(n=max_tasks, random_state=42)

    tasks: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        flow_text = flow_to_text(row)
        true_label = str(row["label_sub"])
        tasks.append(
            {
                "flow_text": flow_text,
                "true_label": true_label,
            }
        )

    random.shuffle(tasks)
    return tasks


# ============= 4. Agent 函数：一个 rollout 对应一个样本 =============

@agl.rollout
def apt_classifier_agent(
    task: Dict[str, Any],
    prompt_template: agl.PromptTemplate,
) -> float:
    """
    Agent Lightning 要求的 rollout 函数。
    输入:
      - task: {"flow_text": ..., "true_label": ...}
      - prompt_template: 会被 APO 不断改写的 Prompt 模板
    输出:
      - reward: 一个 float，当次 roll-out 的得分（0~1）
    """
    # 1. 取出任务信息
    flow_text = task["flow_text"]
    true_label = task["true_label"]  # "APT" or "Benign"

    # 2. 根据当前 prompt_template 渲染出最终 Prompt
    prompt = prompt_template.format(flow_text=flow_text)

    # 3. 调用同步版 OpenAI 客户端
    client = get_sync_client()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
    except Exception:
        # 调用失败时认为是最安全但 reward 最低的情况
        pred_label = "Benign"
        conf = None
    else:
        label_raw, conf = parse_response(content)
        if label_raw is None:
            pred_label = "Benign"
        else:
            pred_label, _ = apply_gate(label_raw, conf, threshold=CONF_THRESHOLD)

    # 4. 根据预测结果和真实标签计算 reward
    #    这里我们：强烈惩罚漏报 APT，稍微惩罚误报 APT
    if true_label == "APT" and pred_label == "APT":
        reward = 1.0          # 抓住 APT，满分
    elif true_label == "APT" and pred_label == "Benign":
        reward = 0.0          # 漏报 APT，最差
    elif true_label == "Benign" and pred_label == "Benign":
        reward = 0.8          # 正确忽略正常流量
    else:  # true Benign, pred APT
        reward = 0.4          # 误报，有点扣分但不至于归零

    return float(reward)


# ============= 5. 主训练流程：配置 APO + Trainer =============

def main():
    # 0) 打开 instrumentation，让 Agent Lightning 能记下 LLM 调用 trace
    instrument_all()

    # 1) 加载环境变量
    load_dotenv()

    # 2) 准备数据集：用 train.csv 做训练 / 验证，保留 test.csv 用作最终评估
    df = pd.read_csv("../data/train.csv", low_memory=False)
    print(f"Loaded TRAIN dataset with {len(df)} rows.")

    tasks = build_tasks_from_df(df, max_tasks=MAX_TASKS)
    print(f"Built {len(tasks)} tasks for training/validation.")

    # 简单按 8:2 划分 train / val
    train_tasks, val_tasks = train_test_split(
        tasks, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Train tasks: {len(train_tasks)}, Val tasks: {len(val_tasks)}")

    # 3) 初始化 APO 算法（使用异步客户端）
    async_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    # 推荐：从环境变量读取，可灵活切换；没有就回落到 gpt-4.1-mini
    gradient_model = os.getenv("APO_GRADIENT_MODEL", "gpt-4.1-mini")
    apply_edit_model = os.getenv("APO_APPLY_EDIT_MODEL", "gpt-4.1-mini")

    algo = agl.APO(
        async_openai_client=async_client,
        gradient_model=gradient_model,
        apply_edit_model=apply_edit_model,
        # 如果你想省钱，还可以顺便把下面这些调小：
        # gradient_batch_size=2,
        # val_batch_size=8,
        # beam_width=2,
        # branch_factor=2,
        # beam_rounds=2,
    )
    # 4) 初始化 Trainer
    trainer = agl.Trainer(
        algorithm=algo,
        n_runners=1,
        strategy="shm",  # 强制使用 SharedMemoryExecutionStrategy（线程模式）
        initial_resources={
            "prompt_template": prompt_template_baseline()
        },
        adapter=agl.TraceToMessages(),
    )

    # 5) 启动训练
    print("Starting Agent Lightning APO training...")
    trainer.fit(
        agent=apt_classifier_agent,
        train_dataset=train_tasks,
        val_dataset=val_tasks,
    )
    print("Training finished.")

    # 6) 从 Store 中导出最终优化后的 prompt_template（而不是从 algo.resources 里取）
    try:
        store = trainer.store  # LightningStoreThreaded / Client 封装

        # 1) 取最新的一版资源快照
        latest = store.get_latest_resources()
        if latest is None:
            print("[WARN] No resources found in store (get_latest_resources() returned None).")
            return

        # 2) latest.resources 是一个 NamedResources（类似字典：name -> resource）
        named_resources = latest.resources
        print("Store resources keys:", list(named_resources.keys()))

        prompt_res = named_resources.get("prompt_template", None)
        if prompt_res is None:
            print("[WARN] 'prompt_template' not found in latest resources.")
            final_template_str = None
        else:
            # 不同版本下，prompt_res 可能是 PromptTemplate 或带 .template 的包装对象
            if hasattr(prompt_res, "template"):
                final_template_str = prompt_res.template
            elif isinstance(prompt_res, agl.PromptTemplate):
                final_template_str = prompt_res.template
            else:
                final_template_str = str(prompt_res)

        if final_template_str:
            print("\n================ FINAL OPTIMIZED PROMPT ================\n")
            print(final_template_str)
            print("\n========================================================\n")

            out_path = "optimized_prompt.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(final_template_str)
            print(f"Final optimized prompt saved to: {out_path}")
        else:
            print("[WARN] No optimized prompt string extracted from store resources.")
    except Exception as e:
        print(f"[WARN] Failed to export optimized prompt from store: {e}")

if __name__ == "__main__":
    main()
