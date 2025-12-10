import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("../data/ZAPT_dataset_V1.csv", low_memory=False)

    # 使用可读标签
    y = df["label_sub"]

    # 简单划分：70% 训练，15% 验证，15% 测试
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        df, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    print("训练集大小:", len(X_train))
    print("验证集大小:", len(X_val))
    print("测试集大小:", len(X_test))

    # 存文件
    X_train.to_csv("../data/train.csv", index=False)
    X_val.to_csv("../data/val.csv", index=False)
    X_test.to_csv("../data/test.csv", index=False)

if __name__ == "__main__":
    main()
