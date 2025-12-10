import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def main():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")

    # 选择部分数值特征
    # features = [
    #     "bidirectional_packets",
    #     "bidirectional_bytes",
    #     "src2dst_packets",
    #     "dst2src_packets",
    #     "bytes_payload_size",
    #     "up_down_ratio",
    #     "down_up_ratio",
    #     "is_server_port_common",
    #     "src2dst_mean_piat_ms",
    #     "dst2src_mean_piat_ms",
    # ]
    features = [
        "bidirectional_packets",
        "bidirectional_bytes",
        "src2dst_packets",
        "dst2src_packets",
        "bytes_payload_size",
        "up_down_ratio",
        "down_up_ratio",
        "src2dst_mean_piat_ms",
        "dst2src_mean_piat_ms",
    ]

    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)

    le = LabelEncoder()
    y_train = le.fit_transform(train["label_sub"])
    y_test = le.transform(test["label_sub"])

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    main()
