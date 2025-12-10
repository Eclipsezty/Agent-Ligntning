import pandas as pd

def main():
    # 1. 读取 CSV（先只读取一部分行，防止太大导致内存占用爆炸）
    csv_path = "../data/ZAPT_dataset_V1.csv"  # 如果你的文件名不一样，这里改一下

    print(f"正在读取数据文件: {csv_path}")
    # 先读前 5000 行做个“试读”
    df_head = pd.read_csv(csv_path, nrows=5000, low_memory=False)
    print(f"预览读取了 {len(df_head)} 行数据（仅用于查看结构）。\n")

    # 2. 查看列名
    print("列名预览：")
    print(df_head.columns.tolist())
    print("\n列总数：", len(df_head.columns))
    print("-" * 80)

    # 3. 尝试识别标签列
    # 你给的样例中有 'label' 和 'label_sub'，我们先看看它们是否存在
    for col in ["label", "label_sub"]:
        if col in df_head.columns:
            print(f"\n列 '{col}' 存在，前几种取值分布：")
            print(df_head[col].value_counts().head(10))
        else:
            print(f"\n列 '{col}' 不存在（没找到这个列名）。")

    print("-" * 80)

    # 4. 如果确认用 'label' 作为主标签，再看一下完整文件的行数（不加载所有列也可以）
    # 注意：这一步可能会慢一点，根据你机器性能和文件大小
    print("正在统计完整文件的行数（只读取索引）...")
    n_rows = sum(1 for _ in open(csv_path, "r", encoding="utf-8", errors="ignore")) - 1
    print(f"完整 CSV 约有 {n_rows} 行数据（不含表头）。")

if __name__ == "__main__":
    main()