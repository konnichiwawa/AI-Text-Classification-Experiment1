import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# =========================
# 1. 读取训练集和测试集
# =========================
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data_unlabeled.csv")

X_train_text = train_df["text"]
y_train = train_df["target"]
X_test_text = test_df["text"]

print("=" * 50)
print("开始训练最终模型")
print(f"训练数据数量：{len(train_df)}")
print(f"测试数据数量：{len(test_df)}")


# =========================
# 2. TF-IDF 特征提取
# =========================
print("\n开始进行 TF-IDF 特征提取...")

vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print(f"训练集 TF-IDF 特征维度：{X_train_tfidf.shape}")
print(f"测试集 TF-IDF 特征维度：{X_test_tfidf.shape}")


# =========================
# 3. 构建最终 MLP 模型
# =========================
print("\n开始训练 MLP 模型...")

mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42
)

mlp.fit(X_train_tfidf, y_train)

print("MLP 模型训练完成！")


# =========================
# 4. 对测试集进行预测
# =========================
print("\n开始对测试集进行预测...")

test_pred = mlp.predict(X_test_tfidf)

print("测试集预测完成！")
print(f"预测结果数量：{len(test_pred)}")
print(f"前20个预测结果：{test_pred[:20]}")


# =========================
# 5. 保存预测结果
# =========================
prediction_df = pd.DataFrame(test_pred)

prediction_df.to_csv(
    "predictions.csv",
    index=False,
    header=False
)

print("\n" + "=" * 50)
print("最终预测文件已保存：predictions_9247.csv")
print(f"预测结果数量：{len(test_pred)}")
print("=" * 50)
