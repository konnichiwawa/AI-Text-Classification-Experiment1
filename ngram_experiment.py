import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# ==============================
# 1. 读取数据
# ==============================
train_df = pd.read_csv("train_data.csv")

X_text = train_df["text"]
y = train_df["target"]


# ==============================
# 2. 使用与前面实验完全相同的数据划分
# ==============================
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# 3. 比较不同的 n-gram 特征表示
# ==============================
configs = [
    ("Unigram", (1, 1)),
    ("Unigram + Bigram", (1, 2))
]


results = []


for name, ngram_range in configs:

    print("=" * 50)
    print(f"开始实验：{name}")
    print(f"ngram_range = {ngram_range}")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=ngram_range
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)

    print(f"训练集特征维度：{X_train_tfidf.shape}")
    print(f"验证集特征维度：{X_val_tfidf.shape}")

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=42
    )

    print("开始训练 MLP...")
    mlp.fit(X_train_tfidf, y_train)

    # 验证
    y_pred = mlp.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"{name} 验证集准确率：{accuracy:.4f}")
    print(f"{name} 验证集准确率：{accuracy * 100:.2f}%")

    results.append({
        "特征表示": name,
        "ngram_range": str(ngram_range),
        "max_features": 8000,
        "hidden_layer_sizes": "(100,)",
        "max_iter": 300,
        "验证集准确率": accuracy
    })


# ==============================
# 4. 保存实验结果
# ==============================
result_df = pd.DataFrame(results)

print("\n" + "=" * 50)
print("实验结果：")
print(result_df)

result_df.to_csv(
    "ngram_experiment_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n实验结果已保存为：ngram_experiment_results.csv")