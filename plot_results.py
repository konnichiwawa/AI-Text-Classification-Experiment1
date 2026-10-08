import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# =========================
# 1. 读取数据
# =========================
train_df = pd.read_csv("train_data.csv")

X_text = train_df["text"]
y = train_df["target"]

# 固定验证集划分
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 2. TF-IDF
# =========================
vectorizer = TfidfVectorizer(max_features=8000)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf = vectorizer.transform(X_val_text)

# =========================
# 3. 训练 MLP
# =========================
print("开始训练 MLP...")

mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42
)

mlp.fit(X_train_tfidf, y_train)

print("MLP 训练完成！")
print(f"实际训练迭代次数: {mlp.n_iter_}")
print(f"最终训练损失: {mlp.loss_curve_[-1]:.6f}")

# =========================
# 4. 绘制专业版训练损失曲线
# =========================
loss = mlp.loss_curve_
iterations = range(1, len(loss) + 1)

plt.figure(figsize=(9, 5.5))

plt.plot(
    iterations,
    loss,
    linewidth=2,
    marker="o",
    markersize=2.5,
    markevery=max(1, len(loss) // 20)
)

# 标记最终点
plt.scatter(
    len(loss),
    loss[-1],
    s=45,
    zorder=3
)

plt.annotate(
    f"Final loss = {loss[-1]:.4f}\nIteration = {len(loss)}",
    xy=(len(loss), loss[-1]),
    xytext=(-110, 45),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", linewidth=1),
    fontsize=9
)

plt.xlabel("Training Iteration", fontsize=11)
plt.ylabel("Training Loss", fontsize=11)

plt.title(
    "MLP Training Loss Curve",
    fontsize=14,
    fontweight="bold"
)

plt.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.5
)

plt.xlim(1, len(loss))

plt.tight_layout()

plt.savefig(
    "mlp_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("专业版 MLP 训练损失曲线已保存为：mlp_training_loss.png")