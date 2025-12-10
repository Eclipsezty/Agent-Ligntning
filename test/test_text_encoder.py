import os
import sys
import pandas as pd

# 把项目下的 src 目录加入 sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from text_encoder import flow_to_text


def main():
    # 1. 读少量数据做测试（防止太大）
    df = pd.read_csv("../data/train.csv", low_memory=False)

    print(f"Train dataset loaded, total rows: {len(df)}")
    print("Preview of label distribution:")
    print(df["label_sub"].value_counts())
    print("-" * 80)

    # 2. 随机抽几条样本
    sample = df.sample(n=5, random_state=42)

    # for idx, row in sample.iterrows():
    #     label = row["label_sub"]
    #     text = flow_to_text(row)
    #
    #     print(f"===== Sample index: {idx} =====")
    #     print(f"True label: {label}")
    #     print("Encoded text:")
    #     print(text)
    #     print("\n")
    for idx, row in sample.iterrows():
        label = row["label_sub"]
        text = flow_to_text(row)

        print(f"===== Sample index: {idx} =====")
        print(f"True label: {label}")

        # 打印几列关键原始字段
        cols_to_show = [
            "protocol_x",
            "application_name",
            "application_category_name",
            "sport",
            "dport",
            "src2dst_bytes",
            "dst2src_bytes",
            "src2dst_packets",
            "dst2src_packets",
            "bidirectional_packets",
            "bidirectional_bytes",
            "bidirectional_duration_ms",
            "up_down_ratio",
            "down_up_ratio",
            "requested_server_name",
            "user_agent",
            "bytes_payload_size",
        ]

        print("Raw fields snapshot:")
        print(row[cols_to_show])
        print("\nEncoded text:")
        print(text)
        print("\n")


if __name__ == "__main__":
    main()
