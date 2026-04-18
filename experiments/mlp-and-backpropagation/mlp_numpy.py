#!/usr/bin/env python3
"""
用 NumPy 从零实现“单层模型 vs 两层 MLP”。

这份脚本故意保持教学风格：

1. 先给一个最简单的单层感知机基线
2. 再给一个两层 MLP
3. 两者都手写前向传播和反向传播
4. 用同一份 XOR 风格的数据做对照

这样你能更直接地看到：
- 单层模型的表达能力上限
- 隐藏层和非线性带来的变化
- 反向传播到底是怎么一步步算的
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def ensure_output_dir() -> None:
    """确保输出目录存在。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_xor_data(
    samples_per_quadrant: int = 80,
    noise_std: float = 0.18,
    seed: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造一个 XOR 风格的二维数据集。

    四个象限各放一簇点：
    - 左下和右上标为 0
    - 左上和右下标为 1

    这类数据的关键特点是：
    不能用一条直线直接分开，所以非常适合用来展示
    “单层线性模型不够，但两层 MLP 可以”的现象。
    """
    rng = np.random.default_rng(seed)

    centers = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    labels = np.array([[0.0], [1.0], [1.0], [0.0]])

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for center, label in zip(centers, labels, strict=True):
        x_part = center + rng.normal(loc=0.0, scale=noise_std, size=(samples_per_quadrant, 2))
        y_part = np.full((samples_per_quadrant, 1), fill_value=float(label[0]))
        x_list.append(x_part)
        y_list.append(y_part)

    x = np.vstack(x_list)
    y = np.vstack(y_list)
    return x, y


def sigmoid(z: np.ndarray) -> np.ndarray:
    """输出层使用 sigmoid，把分数压成 0 到 1 的概率。"""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """二分类交叉熵损失。"""
    epsilon = 1e-8
    y_prob = np.clip(y_prob, epsilon, 1.0 - epsilon)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def compute_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """根据预测概率计算准确率。"""
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return float(np.mean(y_pred == y_true))


class SingleLayerPerceptron:
    """
    最简单的单层感知机。

    它本质上就是：
        z = XW + b
        y_prob = sigmoid(z)

    如果把输出解释成概率，这个模型和逻辑回归是同一类东西。
    如果把 sigmoid 去掉，直接输出连续值，并把损失函数换成 MSE,
    那它又会退化成第 2 周写过的线性回归骨架。

    所以从结构上看：
    - 线性回归：一个线性层，输出连续值
    - 逻辑回归：一个线性层，外面再接一个 sigmoid
    - 两者都可以看成“最简单的单层感知机”
    """

    def __init__(self, input_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(loc=0.0, scale=0.1, size=(input_dim, 1))
        self.bias = 0.0

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        # 单层模型的前向传播只有两步：
        # 1. 线性组合，得到 z
        # 2. 用 sigmoid 把 z 压成概率
        z = x @ self.weight + self.bias
        y_prob = sigmoid(z)
        cache = {"x": x, "z": z, "y_prob": y_prob}
        return y_prob, cache

    def backward(
        self,
        y_true: np.ndarray,
        cache: dict[str, np.ndarray | float],
    ) -> dict[str, np.ndarray | float]:
        x = cache["x"]  # type: ignore[assignment]
        y_prob = cache["y_prob"]  # type: ignore[assignment]
        num_samples = x.shape[0]  # type: ignore[index]

        # 对 sigmoid + BCE，这里会简化成最熟悉的一条：
        # dZ = y_prob - y_true
        #
        # 这条式子和第 2 周逻辑回归里的梯度形式是一致的，
        # 只是这里换了一个“单层感知机”的名字。
        d_z = y_prob - y_true  # type: ignore[operator]
        d_weight = (x.T @ d_z) / num_samples  # type: ignore[operator]
        d_bias = float(np.sum(d_z) / num_samples)
        return {"d_weight": d_weight, "d_bias": d_bias}

    def update(self, gradients: dict[str, np.ndarray | float], learning_rate: float) -> None:
        self.weight -= learning_rate * gradients["d_weight"]  # type: ignore[operator]
        self.bias -= learning_rate * float(gradients["d_bias"])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        y_prob, _ = self.forward(x)
        return y_prob


class TwoLayerMLP:
    """
    一个最小两层 MLP：

        X -> Linear -> Tanh -> Linear -> Sigmoid

    这里的关键不是“层数多了”，而是中间多了非线性激活。
    如果把 Tanh 去掉，这个模型会重新退化成一层线性模型。

    也就是说，多层网络真正增加表达能力的关键不是“多乘了几次矩阵”，
    而是“中间插入了非线性函数”。
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        # 用较小的随机数初始化参数，避免一开始输出过大。
        self.weight1 = rng.normal(loc=0.0, scale=0.6, size=(input_dim, hidden_dim))
        self.bias1 = np.zeros((1, hidden_dim))
        self.weight2 = rng.normal(loc=0.0, scale=0.6, size=(hidden_dim, 1))
        self.bias2 = np.zeros((1, 1))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # 第 1 层线性变换，把输入投影到隐藏空间。
        z1 = x @ self.weight1 + self.bias1

        # 加上非线性之后，模型才能表示弯曲的决策边界。
        a1 = np.tanh(z1)

        # 第 2 层再把隐藏表示映射到输出分数。
        z2 = a1 @ self.weight2 + self.bias2
        y_prob = sigmoid(z2)

        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "y_prob": y_prob,
        }
        return y_prob, cache

    def backward(self, y_true: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
        x = cache["x"]
        a1 = cache["a1"]
        y_prob = cache["y_prob"]
        num_samples = x.shape[0]

        # 第 1 步：从输出层开始。对 sigmoid + BCE，输出层梯度可简化成 y_prob - y_true。
        d_z2 = y_prob - y_true
        d_weight2 = (a1.T @ d_z2) / num_samples
        d_bias2 = np.sum(d_z2, axis=0, keepdims=True) / num_samples

        # 第 2 步：把误差信号传回隐藏层。
        # dA1 表示 loss 对隐藏层激活值 A1 的敏感度。
        # 这里可以理解成：输出层说“哪些地方分错了”，
        # 再把这个纠错信号乘上第二层权重，传回隐藏层。
        d_a1 = d_z2 @ self.weight2.T

        # 第 3 步：穿过 tanh 激活函数。
        # tanh'(z) = 1 - tanh(z)^2，所以这里可以直接用 A1 来算。
        # 这一步就是链式法则在隐藏层上的体现：
        # 先看 loss 对 A1 多敏感，再乘上 A1 对 Z1 的局部导数。
        d_z1 = d_a1 * (1.0 - a1**2)

        # 第 4 步：得到第一层参数的梯度。
        # 到这里，误差信号已经一路传回了第一层。
        d_weight1 = (x.T @ d_z1) / num_samples
        d_bias1 = np.sum(d_z1, axis=0, keepdims=True) / num_samples

        return {
            "d_weight1": d_weight1,
            "d_bias1": d_bias1,
            "d_weight2": d_weight2,
            "d_bias2": d_bias2,
        }

    def update(self, gradients: dict[str, np.ndarray | float], learning_rate: float) -> None:
        self.weight1 -= learning_rate * gradients["d_weight1"]  # type: ignore[operator]
        self.bias1 -= learning_rate * gradients["d_bias1"]  # type: ignore[operator]
        self.weight2 -= learning_rate * gradients["d_weight2"]  # type: ignore[operator]
        self.bias2 -= learning_rate * gradients["d_bias2"]  # type: ignore[operator]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        y_prob, _ = self.forward(x)
        return y_prob


def train_binary_classifier(
    model: SingleLayerPerceptron | TwoLayerMLP,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    epochs: int,
    name: str,
) -> dict[str, list[float] | float]:
    """
    统一的训练循环，同时适配单层模型和两层 MLP。

    这段代码本身也说明了一件事：
    模型结构可以不同，但训练骨架通常是一致的：
    - forward
    - loss
    - backward
    - update
    """
    loss_history: list[float] = []
    accuracy_history: list[float] = []

    for epoch in range(epochs):
        y_prob, cache = model.forward(x)
        loss = binary_cross_entropy(y, y_prob)
        accuracy = compute_accuracy(y, y_prob)

        gradients = model.backward(y, cache)  # type: ignore[arg-type]
        model.update(gradients, learning_rate)

        loss_history.append(loss)
        accuracy_history.append(accuracy)

        if epoch in {0, 1, 2, 4, 9, 19, 49, 99, 199, epochs - 1}:
            print(
                f"[{name}] epoch={epoch + 1:4d}/{epochs} "
                f"loss={loss:.6f} acc={accuracy:.4f}"
            )

    final_accuracy = accuracy_history[-1]
    return {
        "loss_history": loss_history,
        "accuracy_history": accuracy_history,
        "final_accuracy": final_accuracy,
    }


def plot_loss_curves(histories: dict[str, list[float]]) -> Path:
    """画出单层模型和两层 MLP 的 loss 曲线。"""
    plt.figure(figsize=(8, 5))
    for name, loss_history in histories.items():
        plt.plot(loss_history, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.title("Single Layer vs Two-Layer MLP")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "mlp_vs_single_layer_loss.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_decision_boundaries(
    x: np.ndarray,
    y: np.ndarray,
    single_layer_model: SingleLayerPerceptron,
    mlp_model: TwoLayerMLP,
) -> Path:
    """把单层模型和两层 MLP 的决策边界画在同一张图里做对照。"""
    x_min, x_max = x[:, 0].min() - 0.4, x[:, 0].max() + 0.4
    y_min, y_max = x[:, 1].min() - 0.4, x[:, 1].max() + 0.4

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 220),
        np.linspace(y_min, y_max, 220),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    single_probs = single_layer_model.predict_proba(grid).reshape(xx.shape)
    mlp_probs = mlp_model.predict_proba(grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, probs, title in [
        (axes[0], single_probs, "Single-Layer Perceptron"),
        (axes[1], mlp_probs, "Two-Layer MLP"),
    ]:
        contour = ax.contourf(xx, yy, probs, levels=20, cmap="coolwarm", alpha=0.65)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)
        ax.scatter(x[y[:, 0] == 0, 0], x[y[:, 0] == 0, 1], label="Class 0", edgecolors="black", s=28)
        ax.scatter(x[y[:, 0] == 1, 0], x[y[:, 0] == 1, 1], label="Class 1", edgecolors="black", s=28)
        ax.set_title(title)
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")

    fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.92, label="Predicted probability")
    axes[0].legend(loc="upper right")

    output_path = OUTPUT_DIR / "mlp_vs_single_layer_boundary.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    """运行完整的第 3 周实验。"""
    ensure_output_dir()
    x, y = make_xor_data()

    single_layer_model = SingleLayerPerceptron(input_dim=2, seed=0)
    mlp_model = TwoLayerMLP(input_dim=2, hidden_dim=8, seed=1)

    single_history = train_binary_classifier(
        model=single_layer_model,
        x=x,
        y=y,
        learning_rate=0.1,
        epochs=1200,
        name="SingleLayer",
    )
    mlp_history = train_binary_classifier(
        model=mlp_model,
        x=x,
        y=y,
        learning_rate=0.1,
        epochs=1200,
        name="TwoLayerMLP",
    )

    loss_path = plot_loss_curves(
        histories={
            "Single-layer perceptron": single_history["loss_history"],  # type: ignore[arg-type]
            "Two-layer MLP": mlp_history["loss_history"],  # type: ignore[arg-type]
        }
    )
    boundary_path = plot_decision_boundaries(
        x=x,
        y=y,
        single_layer_model=single_layer_model,
        mlp_model=mlp_model,
    )

    print("\nTraining finished.")
    print(f"Single-layer final accuracy: {single_history['final_accuracy']:.4f}")
    print(f"Two-layer MLP final accuracy: {mlp_history['final_accuracy']:.4f}")
    print(f"Saved loss plot: {loss_path}")
    print(f"Saved boundary plot: {boundary_path}")


if __name__ == "__main__":
    main()
