#!/usr/bin/env python3
"""
用 NumPy 从零实现线性回归。

这个脚本故意不依赖任何深度学习框架，目的不是追求工程性能，
而是把训练流程拆开，让你能一眼看到下面四步：

1. 前向计算：根据当前参数做预测
2. 计算损失：预测值和真实值差多少
3. 计算梯度：参数往哪个方向改，loss 才会下降
4. 参数更新：真的把参数往那个方向推一步
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def ensure_output_dir() -> None:
    """确保输出目录存在，方便保存训练曲线和拟合结果图。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_regression_data(
    num_samples: int = 120,
    true_weight: float = 4.0,
    true_bias: float = 7.0,
    noise_std: float = 1.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成一份带噪声的一元线性数据。

    数据真实关系是：
        y = true_weight * x + true_bias + noise

    这里把 x 做成 (N, 1) 的二维数组，而不是 (N,) 的一维数组，
    是为了更贴近机器学习里常见的矩阵表示：

    - N 表示样本数
    - 1 表示特征数
    """
    rng = np.random.default_rng(seed)

    # 把输入范围控制在 [-2, 2]，便于用不同学习率做对比实验。
    x = rng.uniform(-2.0, 2.0, size=(num_samples, 1))
    noise = rng.normal(loc=0.0, scale=noise_std, size=(num_samples, 1))
    y = true_weight * x + true_bias + noise
    return x, y


def predict(x: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    """
    线性回归的前向计算。

    对一元线性回归来说，公式是：
        y_pred = x * w + b

    这里写成矩阵形式：
        y_pred = X @ W + b

    这样以后扩展到多特征时，代码几乎不需要改。
    """
    return x @ weight + bias


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差 MSE。

    为什么要平方？
    - 正负误差不会互相抵消
    - 大误差会被放大，模型会更重视偏差大的样本
    """
    return float(np.mean((y_true - y_pred) ** 2))


def compute_gradients(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    计算 MSE 对参数的梯度。

    设样本数为 N，损失函数为：
        L = mean((y_pred - y_true)^2)

    因为：
        y_pred = X @ W + b

    所以梯度可以直接写成：
        dL/dW = (2 / N) * X^T @ (y_pred - y_true)
        dL/db = (2 / N) * sum(y_pred - y_true)

    这两条式子非常重要，它们说明了：
    - 如果预测值整体偏大，梯度会推动参数往减小预测的方向走
    - 如果预测值整体偏小，梯度会推动参数往增大预测的方向走
    """
    num_samples = x.shape[0]
    error = y_pred - y_true
    d_weight = (2.0 / num_samples) * (x.T @ error)
    d_bias = float((2.0 / num_samples) * np.sum(error))
    return d_weight, d_bias


def train_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    epochs: int,
) -> dict[str, np.ndarray | float | list[float]]:
    """
    用最基础的梯度下降训练线性回归。

    这里故意不用类，也不做太多抽象，因为教学阶段最重要的是
    把训练步骤看得清楚，而不是把代码封装得很漂亮。
    """
    # 权重初始化为 0，表示模型一开始什么都没学到。
    weight = np.zeros((x.shape[1], 1))
    bias = 0.0
    loss_history: list[float] = []

    for epoch in range(epochs):
        # 第 1 步：前向计算，得到当前参数下的预测结果。
        y_pred = predict(x, weight, bias)

        # 第 2 步：计算当前 loss。
        loss = mean_squared_error(y, y_pred)
        loss_history.append(loss)

        # 第 3 步：根据 loss 对参数的敏感度，计算梯度。
        d_weight, d_bias = compute_gradients(x, y, y_pred)

        # 第 4 步：沿着“梯度的反方向”更新参数。
        # 因为梯度指向的是“loss 增加最快”的方向，
        # 所以减去梯度，等价于朝“loss 下降”的方向走。
        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias

        # 打印稀疏日志，帮助你看到训练的过程。
        if epoch in {0, 1, 2, 4, 9, 19, 49, 99, epochs - 1}:
            print(
                f"[Linear][lr={learning_rate:.4f}] "
                f"epoch={epoch + 1:4d}/{epochs} "
                f"loss={loss:.6f} "
                f"w={float(weight[0, 0]):.4f} "
                f"b={bias:.4f}"
            )

    return {
        "weight": weight,
        "bias": bias,
        "loss_history": loss_history,
        "final_predictions": predict(x, weight, bias),
    }


def plot_regression_fit(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    bias: float,
) -> Path:
    """
    画出原始数据点和训练完成后的拟合直线。

    这张图的意义是把“loss 在下降”翻译成更直观的结果：
    模型学到的直线是否真的贴近数据分布。
    """
    sort_index = np.argsort(x[:, 0])
    x_sorted = x[sort_index]
    y_pred_sorted = predict(x_sorted, weight, bias)

    plt.figure(figsize=(8, 5))
    plt.scatter(x[:, 0], y[:, 0], alpha=0.75, label="Training data")
    plt.plot(x_sorted[:, 0], y_pred_sorted[:, 0], color="crimson", linewidth=2.5, label="Model fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "linear_regression_fit.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_loss_histories(histories: dict[float, list[float]]) -> Path:
    """
    把不同学习率下的 loss 曲线画到同一张图上。

    学习率是初学者最应该先建立直觉的超参数之一：
    - 太小：每一步都走得很保守，loss 降得慢
    - 合适：下降快而且稳
    - 太大：可能在最优点附近来回震荡，甚至直接发散
    """
    plt.figure(figsize=(8, 5))
    for learning_rate, loss_history in histories.items():
        plt.plot(loss_history, label=f"lr={learning_rate}")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Linear Regression: Learning Rate Comparison")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "linear_regression_learning_rates.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    """运行完整的线性回归教学实验。"""
    ensure_output_dir()

    x, y = make_regression_data()

    # 这三个学习率分别对应：
    # - 0.01：偏小，收敛慢
    # - 0.05：通常比较合适
    # - 0.80：偏大，可能出现明显震荡
    learning_rates = [0.01, 0.05, 0.80]
    epochs = 120

    histories: dict[float, list[float]] = {}
    best_result: dict[str, np.ndarray | float | list[float]] | None = None
    best_loss = float("inf")

    for learning_rate in learning_rates:
        result = train_linear_regression(
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

    fit_path = plot_regression_fit(
        x=x,
        y=y,
        weight=best_result["weight"],  # type: ignore[arg-type]
        bias=float(best_result["bias"]),
    )
    loss_path = plot_loss_histories(histories)

    learned_weight = float(best_result["weight"][0, 0])  # type: ignore[index]
    learned_bias = float(best_result["bias"])
    print("\nLinear regression training finished.")
    print(f"Best final loss: {best_loss:.6f}")
    print(f"Learned weight: {learned_weight:.4f}")
    print(f"Learned bias:   {learned_bias:.4f}")
    print(f"Saved fit plot: {fit_path}")
    print(f"Saved loss plot: {loss_path}")


if __name__ == "__main__":
    main()
