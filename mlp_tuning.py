import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# 1. 加载数据
# ============================================================
train_df = pd.read_csv("train_data.csv")

X_text = train_df["text"]
y = train_df["target"]

# 固定随机种子，保证实验可以复现
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("=" * 60)
print("MLP 参数实验")
print("=" * 60)
print(f"训练集数量: {len(X_train_text)}")
print(f"验证集数量: {len(X_val_text)}")

# ============================================================
# 2. TF-IDF
# ============================================================
vectorizer = TfidfVectorizer(max_features=8000)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf = vectorizer.transform(X_val_text)

print(f"TF-IDF 训练集: {X_train_tfidf.shape}")
print(f"TF-IDF 验证集: {X_val_tfidf.shape}")

# ============================================================
# 3. MLP 参数实验
# ============================================================
configs = [
    ("MLP (50,)", (50,)),
    ("MLP (100,)", (100,)),
    ("MLP (200,)", (200,)),
    ("MLP (100,100)", (100, 100)),
]

results = []

for name, hidden_layers in configs:

    print("\n" + "-" * 60)
    print(f"开始训练: {name}")

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=300,
        random_state=42
    )

    mlp.fit(X_train_tfidf, y_train)

    y_pred = mlp.predict(X_val_tfidf)

    accuracy = accuracy_score(y_val, y_pred)

    results.append({
        "模型": name,
        "hidden_layer_sizes": str(hidden_layers),
        "max_iter": 300,
        "验证集准确率": accuracy
    })

    print(f"{name} 验证集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# ============================================================
# 4. 汇总结果
# ============================================================
result_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MLP 参数实验结果")
print("=" * 60)

print(result_df.to_string(index=False))

# 保存实验结果
result_df.to_csv(
    "mlp_tuning_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n实验结果已保存到: mlp_tuning_results.csv")