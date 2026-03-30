#!/usr/bin/env python3
"""
用 NumPy 从零实现二分类逻辑回归。

逻辑回归可以理解成：

1. 先做一次线性变换：z = X @ W + b
2. 再通过 sigmoid 把 z 压缩到 0 到 1
3. 把输出值解释为“属于正类的概率”

如果你把它和线性回归对照着看，会更容易抓住核心差异：

- 线性回归：输出连续值，常配 MSE
- 逻辑回归：输出类别概率，常配交叉熵
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# 使用无界面后端，避免在纯终端环境里因为 display 问题报错。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def ensure_output_dir() -> None:
    """确保保存图片的目录存在。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_classification_data(
    samples_per_class: int = 120,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成一个二维二分类数据集。

    这里人为构造两个高斯簇：
    - 类别 0 大致分布在左下方
    - 类别 1 大致分布在右上方

    选择二维数据是因为它可以很直观地画出分类边界。
    """
    rng = np.random.default_rng(seed)

    class_0 = rng.normal(loc=(-2.0, -1.6), scale=1.0, size=(samples_per_class, 2))
    class_1 = rng.normal(loc=(2.0, 1.8), scale=1.1, size=(samples_per_class, 2))

    x = np.vstack([class_0, class_1])
    y = np.vstack(
        [
            np.zeros((samples_per_class, 1)),
            np.ones((samples_per_class, 1)),
        ]
    )
    return x, y


def standardize_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对特征做标准化。

    为什么这里要标准化？
    - 两个维度如果尺度差异很大，梯度下降会更难调学习率
    - 标准化之后，训练通常更稳定

    返回值包括：
    - 标准化后的特征
    - 均值
    - 标准差

    后两者保留下来，是为了强调：
    训练时对输入做的变换，在推理时也必须保持一致。
    """
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    x_standardized = (x - mean) / std
    return x_standardized, mean, std


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    sigmoid 把任意实数映射到 0 到 1。

    它的输出可以解释成概率，这是逻辑回归能做分类的关键。

    这里先对 z 做裁剪，是为了避免 exp 溢出。
    """
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(x: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    """
    输出每个样本属于正类的概率。

    公式：
        p(y=1|x) = sigmoid(X @ W + b)
    """
    linear_output = x @ weight + bias
    return sigmoid(linear_output)


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    二元交叉熵损失。

    公式：
        L = -mean(y * log(p) + (1-y) * log(1-p))

    它惩罚的是“自信但错误”的预测。
    例如：
    - 真标签是 1，但模型给出 0.01
    - 真标签是 0，但模型给出 0.99

    这两种情况的惩罚都会很大。
    """
    epsilon = 1e-8
    y_prob = np.clip(y_prob, epsilon, 1.0 - epsilon)
    loss = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(loss)


def compute_gradients(
    x: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    计算逻辑回归在交叉熵损失下的梯度。

    这是一个非常值得记住的结果：

        dL/dW = X^T @ (y_prob - y_true) / N
        dL/db = sum(y_prob - y_true) / N

    你会发现它和线性回归的梯度结构很像，只是误差项从
    “连续值预测误差”变成了“概率预测误差”。
    """
    num_samples = x.shape[0]
    error = y_prob - y_true
    d_weight = (x.T @ error) / num_samples
    d_bias = float(np.sum(error) / num_samples)
    return d_weight, d_bias


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    epochs: int,
) -> dict[str, np.ndarray | float | list[float]]:
    """
    用最基础的梯度下降训练逻辑回归。

    和线性回归相比，训练循环并没有本质变化：
    - 前向
    - 算 loss
    - 算梯度
    - 更新参数

    真正的差别来自：
    - 输出层从线性输出改成了 sigmoid 概率
    - 损失函数从 MSE 改成了交叉熵
    """
    weight = np.zeros((x.shape[1], 1))
    bias = 0.0
    loss_history: list[float] = []

    for epoch in range(epochs):
        y_prob = predict_proba(x, weight, bias)
        loss = binary_cross_entropy(y, y_prob)
        loss_history.append(loss)

        d_weight, d_bias = compute_gradients(x, y, y_prob)

        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias

        if epoch in {0, 1, 2, 4, 9, 19, 49, 99, epochs - 1}:
            accuracy = compute_accuracy(y, y_prob)
            print(
                f"[Logistic][lr={learning_rate:.4f}] "
                f"epoch={epoch + 1:4d}/{epochs} "
                f"loss={loss:.6f} "
                f"acc={accuracy:.4f} "
                f"w={weight.ravel()} "
                f"b={bias:.4f}"
            )

    return {
        "weight": weight,
        "bias": bias,
        "loss_history": loss_history,
        "final_probabilities": predict_proba(x, weight, bias),
    }


def predict_label(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    把概率转成最终类别标签。

    默认阈值是 0.5：
    - 大于等于 0.5 判为 1
    - 小于 0.5 判为 0

    这个阈值不是绝对固定的，实际任务中可以根据业务需求调整。
    """
    return (y_prob >= threshold).astype(np.int32)


def compute_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """根据预测概率计算分类准确率。"""
    y_pred = predict_label(y_prob)
    return float(np.mean(y_pred == y_true))


def plot_dataset_and_boundary(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    bias: float,
) -> Path:
    """
    画出样本点和分类边界。

    在二维情况下，逻辑回归的分类边界满足：
        w1 * x1 + w2 * x2 + b = 0

    也就是一条直线。
    这张图能帮你把“概率分类器”直观地理解成：
    模型在特征空间里寻找一条把两类分开的边界。
    """
    x0 = x[y[:, 0] == 0]
    x1 = x[y[:, 0] == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x0[:, 0], x0[:, 1], alpha=0.7, label="Class 0")
    plt.scatter(x1[:, 0], x1[:, 1], alpha=0.7, label="Class 1")

    w1 = float(weight[0, 0])
    w2 = float(weight[1, 0])

    # 如果 w2 非常接近 0，边界会接近竖直线；这里做个保护。
    if abs(w2) > 1e-8:
        x_values = np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 200)
        y_values = -(w1 * x_values + bias) / w2
        plt.plot(x_values, y_values, color="black", linewidth=2.5, label="Decision boundary")

    plt.xlabel("feature 1 (standardized)")
    plt.ylabel("feature 2 (standardized)")
    plt.title("Logistic Regression Decision Boundary")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "logistic_regression_boundary.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_loss_histories(histories: dict[float, list[float]]) -> Path:
    """画出不同学习率下的交叉熵损失曲线。"""
    plt.figure(figsize=(8, 5))
    for learning_rate, loss_history in histories.items():
        plt.plot(loss_history, label=f"lr={learning_rate}")

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.title("Logistic Regression: Learning Rate Comparison")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "logistic_regression_learning_rates.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    """运行完整的逻辑回归教学实验。"""
    ensure_output_dir()

    x_raw, y = make_classification_data()
    x, mean, std = standardize_features(x_raw)

    # 三个学习率分别体现：
    # - 0.05：能学，但速度偏慢
    # - 0.30：通常比较合适
    # - 2.00：偏大，可能出现震荡
    learning_rates = [0.05, 0.30, 2.00]
    epochs = 150

    histories: dict[float, list[float]] = {}
    best_result: dict[str, np.ndarray | float | list[float]] | None = None
    best_loss = float("inf")

    for learning_rate in learning_rates:
        result = train_logistic_regression(
            x=x,
            y=y,
            learning_rate=learning_rate,
            epochs=epochs,
        )

        final_loss = float(result["loss_history"][-1])  # type: ignore[index]
        histories[learning_rate] = result["loss_history"]  # type: ignore[assignment]

        if final_loss < best_loss:
            best_loss = final_loss
            best_result = result

    assert best_result is not None

    boundary_path = plot_dataset_and_boundary(
        x=x,
        y=y,
        weight=best_result["weight"],  # type: ignore[arg-type]
        bias=float(best_result["bias"]),
    )
    loss_path = plot_loss_histories(histories)

    final_probabilities = best_result["final_probabilities"]  # type: ignore[assignment]
    final_accuracy = compute_accuracy(y, final_probabilities)  # type: ignore[arg-type]
    learned_weight = best_result["weight"]  # type: ignore[assignment]
    learned_bias = float(best_result["bias"])

    print("\nLogistic regression training finished.")
    print(f"Feature mean used for standardization: {mean.ravel()}")
    print(f"Feature std used for standardization:  {std.ravel()}")
    print(f"Best final loss: {best_loss:.6f}")
    print(f"Final accuracy:  {final_accuracy:.4f}")
    print(f"Learned weight:  {learned_weight.ravel()}")
    print(f"Learned bias:    {learned_bias:.4f}")
    print(f"Saved boundary plot: {boundary_path}")
    print(f"Saved loss plot: {loss_path}")


if __name__ == "__main__":
    main()
