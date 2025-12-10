import os

from dotenv import load_dotenv
from openai import OpenAI

def main():
    # 1. 读取 .env 中的环境变量
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未配置，请先在 .env 文件中设置")

    # 2. 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # 3. 发起一次最简单的对话
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "请用一句话告诉我：我这个 Python 环境已经可以调用大模型了吗？"}
        ],
    )

    print("模型返回：")
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
