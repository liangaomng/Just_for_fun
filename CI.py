import numpy as np
import matplotlib.pyplot as plt
import scipy
# 假设我们有一些预测的数据和真实的数据
np.random.seed(0)  # 为了重现性
true_y = np.linspace(0.8, 1.2, 100)  # 真实的y值，简单线性关系
predicted_y = true_y + np.random.normal(0, 0.02, true_y.shape)  # 预测的y值，添加一些噪声

# 现在我们将模拟不同的数据集大小效果，这里简单使用子集
sizes = [50, 25, 10]  # 不同的样本大小
plt.figure(figsize=(15, 5))

for i, size in enumerate(sizes, 1):
    # 随机选择一些数据点作为示例
    indices = np.random.choice(len(true_y), size, replace=False)
    sample_true_y = true_y[indices]
    sample_predicted_y = predicted_y[indices]

    # 绘制散点图
    plt.subplot(1, len(sizes), i)
    plt.scatter(sample_true_y, sample_predicted_y, alpha=0.5, label=f'{size} samples')

    # 绘制y=x线，表示完美预测
    plt.plot([0.8, 1.2], [0.8, 1.2], 'r--', label='Perfect Prediction')

    # 设置图的标题和图例
    plt.title(f'Extraction: {size} examples')
    plt.xlabel('True y')
    plt.ylabel('Predicted y')
    plt.legend()
    plt.grid(True)

# 展示图表
plt.tight_layout()
plt.show()


# 基于上面的代码，我们现在将添加表示95%置信区间的阴影区域

# 函数来计算置信区间
def calculate_confidence_interval(y_preds, confidence=0.95):
    # 根据预测值的标准误差计算置信区间
    std_error = np.std(y_preds) / np.sqrt(len(y_preds))
    h = std_error * scipy.stats.t.ppf((1 + confidence) / 2., len(y_preds) - 1)
    return h


# 继续使用上面生成的数据
sizes = [50, 25, 10]  # 不同的样本大小
plt.figure(figsize=(15, 5))

for i, size in enumerate(sizes, 1):
    # 随机选择一些数据点作为示例
    indices = np.random.choice(len(true_y), size, replace=False)
    sample_true_y = true_y[indices]
    sample_predicted_y = predicted_y[indices]

    # 计算置信区间
    confidence_interval = calculate_confidence_interval(sample_predicted_y, confidence=0.95)

    # 绘制散点图
    plt.subplot(1, len(sizes), i)
    plt.scatter(sample_true_y, sample_predicted_y, alpha=0.5, label=f'{size} samples')

    # 绘制y=x线，表示完美预测
    plt.plot([0.8, 1.2], [0.8, 1.2], 'r--', label='Perfect Prediction')

    # 绘制置信区间
    plt.fill_between([0.8, 1.2],
                     [0.8 - confidence_interval, 1.2 - confidence_interval],
                     [0.8 + confidence_interval, 1.2 + confidence_interval],
                     color='blue', alpha=0.2, label='95% Confidence Interval')

    # 设置图的标题和图例
    plt.title(f'Extraction: {size} examples')
    plt.xlabel('True y')
    plt.ylabel('Predicted y')
    plt.legend()
    plt.grid(True)

# 展示图表
plt.tight_layout()
plt.show()

