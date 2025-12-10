from openai import OpenAI
from dotenv import load_dotenv

# 你想重点测试的模型，可以按需增删
CANDIDATE_CHAT_MODELS = [
    "gpt-5.1",      # 最高端（如果有）
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o-mini",  # 官方推荐小模型
]

EMBEDDING_MODEL = "text-embedding-3-small"
IMAGE_MODEL = "gpt-image-1"


def list_models(client):
    """列出模型，返回一个 set，方便后面判断支持哪些模型"""
    print("  - 列出模型...")
    models = client.models.list()
    ids = [m.id for m in models.data]
    print(f"    共 {len(ids)} 个模型（仅展示前 10 个）：")
    for mid in ids[:10]:
        print("     •", mid)
    return set(ids)


def test_chat_model(client, model_id):
    """测试一个 chat 模型能否正常调用"""
    print(f"    测试 Chat 模型: {model_id}")
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "简单介绍一下你自己，控制在一句话。"}],
            max_tokens=50,
        )
        text = resp.choices[0].message.content.strip()
        print(f"      ✅ 成功：{text}")
        return True
    except Exception as e:
        print(f"      ❌ 失败：{repr(e)}")
        return False


def test_embedding(client, model_id):
    print(f"  - 测试 Embedding 模型: {model_id}")
    try:
        resp = client.embeddings.create(
            model=model_id,
            input="这是一个用于测试的句子。"
        )
        dim = len(resp.data[0].embedding)
        print(f"    ✅ 成功：向量维度 = {dim}")
        return True
    except Exception as e:
        print(f"    ❌ 失败：{repr(e)}")
        return False


def test_image(client, model_id):
    print(f"  - 测试 图片生成 模型: {model_id}")
    try:
        resp = client.images.generate(
            model=model_id,
            prompt="a simple blue cube icon",
            n=1,
            size="512x512"
        )
        # 这里只打印一下有没有返回，而不保存图片
        print("    ✅ 成功：返回了图片结果（已省略具体内容）")
        return True
    except Exception as e:
        print(f"    ❌ 失败：{repr(e)}")
        return False


def full_check(name, api_key):
    print(f"\n==================== 测试 {name} ====================")
    client = OpenAI(api_key=api_key)

    # 1. 列出模型
    try:
        model_ids = list_models(client)
    except Exception as e:
        print("  ❌ 无法列出模型，可能是 key 权限异常：", repr(e))
        return

    # 2. 逐个测试候选 Chat 模型（前提是这个 key 的确有这个模型）
    print("  - 测试 Chat 功能：")
    for mid in CANDIDATE_CHAT_MODELS:
        if mid in model_ids:
            test_chat_model(client, mid)
        else:
            print(f"    ⚠ 跳过 {mid}（该 key 不支持该模型）")

    # 3. 测试 Embedding
    if EMBEDDING_MODEL in model_ids:
        test_embedding(client, EMBEDDING_MODEL)
    else:
        print(f"  ⚠ 跳过 Embedding：{EMBEDDING_MODEL} 不在可用列表中")

    # 4. 测试 图片生成
    if IMAGE_MODEL in model_ids:
        test_image(client, IMAGE_MODEL)
    else:
        print(f"  ⚠ 跳过 图片生成：{IMAGE_MODEL} 不在可用列表中")

    print(f"================== {name} 测试结束 ==================\n")


if __name__ == "__main__":

    xhb_key = os.getenv("OPENAI_API_KEY_xhb")
    orgt_key = os.getenv("OPENAI_API_KEY")

    full_check("xhb_key", xhb_key)
    full_check("orgt_key", orgt_key)
