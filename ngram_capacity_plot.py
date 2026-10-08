import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# 1. 准备实验结果
# ==============================

results = pd.DataFrame({
    "实验设置": [
        "Unigram\n8000",
        "Unigram + Bigram\n8000",
        "Unigram + Bigram\n12000"
    ],
    "验证集准确率": [
        0.9240,
        0.9172,
        0.9247
    ]
})


# ==============================
# 2. 绘制柱状图
# ==============================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    results["实验设置"],
    results["验证集准确率"] * 100
)

# 添加准确率数值
for bar, value in zip(bars, results["验证集准确率"] * 100):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.1,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )


# ==============================
# 3. 设置图表
# ==============================

plt.ylabel("Validation Accuracy (%)")
plt.xlabel("Feature Representation")

plt.title("Effect of N-gram Representation and Feature Capacity")

plt.ylim(90, 93)

plt.tight_layout()


# ==============================
# 4. 保存图片
# ==============================

plt.savefig(
    "ngram_capacity_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("图片已保存为：ngram_capacity_comparison.png")