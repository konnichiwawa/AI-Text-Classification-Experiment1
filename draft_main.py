# -*- coding: utf-8 -*-

# ==========================================
# 实验一：最终模型训练与测试集预测
#
# 最佳模型：
# MLP
#
# TF-IDF：
# max_features = 8000
#
# MLP：
# hidden_layer_sizes = (100,)
# max_iter = 300
# random_state = 42
# ==========================================


# ==========================================
# 1. 导入库
# ==========================================

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# ==========================================
# 2. 加载训练数据和测试数据
# ==========================================

train_df = pd.read_csv(
    'train_data.csv'
)

test_df = pd.read_csv(
    'test_data_unlabeled.csv'
)


# 训练文本
X_train = train_df['text'].astype(str).tolist()

# 训练标签
y_train = train_df['target'].values

# 测试文本
X_test = test_df['text'].astype(str).tolist()


print("=" * 60)
print("实验一：最终模型训练与测试集预测")
print("=" * 60)

print(
    f"训练数据数量: {len(X_train)}"
)

print(
    f"测试数据数量: {len(X_test)}"
)


# ==========================================
# 3. TF-IDF 特征提取
# ==========================================

print("\n")
print("=" * 60)
print("1. TF-IDF 特征提取")
print("=" * 60)

vectorizer = TfidfVectorizer(
    max_features=8000
)


# 在全部训练数据上建立词汇表
X_train_tfidf = vectorizer.fit_transform(
    X_train
)


# 使用训练数据得到的词汇表
# 转换测试数据
X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    f"训练集 TF-IDF 矩阵: "
    f"{X_train_tfidf.shape}"
)

print(
    f"测试集 TF-IDF 矩阵: "
    f"{X_test_tfidf.shape}"
)


# ==========================================
# 4. 创建最佳 MLP 模型
# ==========================================

print("\n")
print("=" * 60)
print("2. 训练最终 MLP 模型")
print("=" * 60)

mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42
)


print(
    "--- 开始使用全部训练数据训练 MLP ---"
)

mlp_classifier.fit(
    X_train_tfidf,
    y_train
)

print(
    "最终模型训练完成！"
)


# ==========================================
# 5. 对测试集进行预测
# ==========================================

print("\n")
print("=" * 60)
print("3. 测试集预测")
print("=" * 60)

y_pred_test = mlp_classifier.predict(
    X_test_tfidf
)

print(
    "测试集预测完成！"
)


# ==========================================
# 6. 保存 predictions.csv
# ==========================================

print("\n")
print("=" * 60)
print("4. 保存预测结果")
print("=" * 60)

prediction_df = pd.DataFrame(
    y_pred_test
)

prediction_df.to_csv(
    'predictions.csv',
    index=False,
    header=False
)

print(
    "predictions.csv 保存成功！"
)

print(
    f"预测结果数量: {len(y_pred_test)}"
)

print(
    "文件位置：当前实验文件夹 / predictions.csv"
)


# ==========================================
# 7. 显示部分预测结果
# ==========================================

print("\n")
print("前20个预测结果：")

print(
    y_pred_test[:20]
)


print("\n")
print("=" * 60)
print("最终预测完成！")
print("=" * 60)