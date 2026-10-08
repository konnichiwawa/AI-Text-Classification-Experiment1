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
# 3. 使用 Unigram + Bigram
#    测试更大的特征容量
# ==============================
print("=" * 50)
print("开始实验：Unigram + Bigram (12000)")
print("ngram_range = (1, 2)")
print("max_features = 12000")


# ==============================
# 4. TF-IDF
# ==============================
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf = vectorizer.transform(X_val_text)

print(f"训练集特征维度：{X_train_tfidf.shape}")
print(f"验证集特征维度：{X_val_tfidf.shape}")


# ==============================
# 5. MLP
# ==============================
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42
)

print("开始训练 MLP...")
mlp.fit(X_train_tfidf, y_train)


# ==============================
# 6. 验证集评估
# ==============================
y_pred = mlp.predict(X_val_tfidf)

accuracy = accuracy_score(y_val, y_pred)

print(f"Unigram + Bigram (12000) 验证集准确率：{accuracy:.4f}")
print(f"Unigram + Bigram (12000) 验证集准确率：{accuracy * 100:.2f}%")


# ==============================
# 7. 保存实验结果
# ==============================
result_df = pd.DataFrame([{
    "特征表示": "Unigram + Bigram",
    "ngram_range": "(1, 2)",
    "max_features": 12000,
    "hidden_layer_sizes": "(100,)",
    "max_iter": 300,
    "验证集准确率": accuracy
}])

print("\n" + "=" * 50)
print("实验结果：")
print(result_df)

result_df.to_csv(
    "ngram_capacity_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n实验结果已保存为：ngram_capacity_results.csv")