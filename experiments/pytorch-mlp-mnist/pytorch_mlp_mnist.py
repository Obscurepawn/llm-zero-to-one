#!/usr/bin/env python3
"""
用 PyTorch 重做 MLP，并在 MNIST 上训练。

这份脚本有两个目标：

1. 帮你把第 2 周、第 3 周手写过的前向传播 / loss / 反向传播
   和 PyTorch 的标准训练流程一一对上。
2. 帮你建立最基本的训练工程直觉：
   - Dataset / DataLoader 在做什么
   - train / val / test 为什么要分开
   - dropout / weight decay 为什么有助于泛化

默认会训练两个版本：
- baseline: 不加正则化
- regularized: 加 dropout 和 weight decay
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message="CUDA initialization:.*")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset, random_split
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "当前环境缺少 torch 或 torchvision。请先执行 "
        "`python -m pip install torch torchvision matplotlib` 再运行本脚本。"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
DATA_DIR = ROOT_DIR / "data"


@dataclass
class ExperimentConfig:
    """保存一组训练配置，方便并排比较 baseline 和 regularized。"""

    name: str
    hidden_dims: tuple[int, int]
    learning_rate: float
    batch_size: int
    epochs: int
    dropout: float
    weight_decay: float


def ensure_directories() -> None:
    """确保数据目录和输出目录存在。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """固定随机种子，尽量保证每次运行结果接近。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    教学实验默认优先保证可运行性。

    当前环境里即使装了带 CUDA 的 torch，也不一定真的具备可用驱动，
    所以这里默认直接走 CPU，避免把初学者的注意力带到环境问题上。
    """
    return torch.device("cpu")


def build_transform() -> transforms.Compose:
    """
    定义 MNIST 预处理流程。

    - ToTensor: 把图片转成张量，范围从 0-255 变成 0-1
    - Normalize: 用 MNIST 常见均值和标准差做标准化
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def limit_subset(dataset: Dataset, max_items: int | None, seed: int) -> Dataset:
    """
    从大数据集中截出一个固定大小的子集，降低教学实验的运行成本。
    """
    if max_items is None or max_items >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_items].tolist()
    return Subset(dataset, indices)


def load_mnist_datasets(
    val_size: int = 5_000,
    train_subset: int = 12_000,
    test_subset: int = 2_000,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    读取并切分 MNIST。

    返回：
    - train_dataset
    - val_dataset
    - test_dataset
    """
    transform = build_transform()

    try:
        full_train = datasets.MNIST(
            root=DATA_DIR,
            train=True,
            transform=transform,
            download=True,
        )
        test_dataset = datasets.MNIST(
            root=DATA_DIR,
            train=False,
            transform=transform,
            download=True,
        )
    except RuntimeError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "MNIST 下载失败。当前环境可能限制了网络访问。"
            "如果你在受限环境里运行，请先允许数据下载，或者手动把 MNIST 放到 "
            f"`{DATA_DIR}` 再重新运行。"
        ) from exc

    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataset = limit_subset(train_dataset, train_subset, seed=seed)
    val_dataset = limit_subset(val_dataset, min(val_size, 2_000), seed=seed + 1)
    test_dataset = limit_subset(test_dataset, test_subset, seed=seed + 2)

    return train_dataset, val_dataset, test_dataset


def make_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    把数据集封装成 DataLoader。

    训练集通常要打乱，验证集和测试集不需要。
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class MLPClassifier(nn.Module):
    """
    一个用于 MNIST 的两层 MLP。

    对应结构是：
        28x28 image
        -> flatten
        -> Linear
        -> ReLU
        -> Dropout
        -> Linear
        -> ReLU
        -> Linear
        -> logits

    最后一层不手动做 softmax，因为 CrossEntropyLoss 内部已经包含了它。
    """

    def __init__(self, hidden_dims: tuple[int, int], dropout: float) -> None:
        super().__init__()
        hidden_dim1, hidden_dim2 = hidden_dims

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """根据 logits 计算当前 batch 的准确率。"""
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在验证集或测试集上评估平均 loss 和准确率。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    训练一个 epoch。

    - forward
    - loss
    - zero_grad
    - backward
    - step
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy


def train_experiment(
    config: ExperimentConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
) -> dict[str, object]:
    """训练一组实验配置，并返回完整历史记录。"""
    train_loader, val_loader, test_loader = make_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.batch_size,
    )

    model = MLPClassifier(hidden_dims=config.hidden_dims, dropout=config.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    val_loss_history: list[float] = []
    val_accuracy_history: list[float] = []

    print(f"\n[Experiment] {config.name}")
    print(
        "config="
        f"hidden_dims={config.hidden_dims}, "
        f"lr={config.learning_rate}, "
        f"batch_size={config.batch_size}, "
        f"epochs={config.epochs}, "
        f"dropout={config.dropout}, "
        f"weight_decay={config.weight_decay}"
    )

    for epoch in range(config.epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        print(
            f"[{config.name}] epoch={epoch + 1:2d}/{config.epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )

    test_loss, test_accuracy = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    print(f"[{config.name}] test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}")

    return {
        "config": config,
        "model": model,
        "train_loss_history": train_loss_history,
        "train_accuracy_history": train_accuracy_history,
        "val_loss_history": val_loss_history,
        "val_accuracy_history": val_accuracy_history,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }


def unnormalize_image(image: torch.Tensor) -> np.ndarray:
    """把标准化后的图片转回便于显示的 0-1 范围。"""
    image = image.cpu() * 0.3081 + 0.1307
    image = torch.clamp(image, 0.0, 1.0)
    return image.squeeze(0).numpy()


def plot_sample_grid(dataset: Dataset, output_path: Path, seed: int = 0) -> None:
    """从数据集中抽几张图片，帮助理解输入长什么样。"""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=12, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for ax, index in zip(axes.ravel(), indices, strict=True):
        image, label = dataset[index]
        ax.imshow(unnormalize_image(image), cmap="gray")
        ax.set_title(f"label={label}")
        ax.axis("off")

    plt.suptitle("MNIST Sample Grid", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_training_curves(results: list[dict[str, object]], output_path: Path) -> None:
    """把两个实验的训练 loss 和验证准确率画到同一张图上。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for result in results:
        config = result["config"]
        assert isinstance(config, ExperimentConfig)

        axes[0].plot(result["train_loss_history"], label=config.name)  # type: ignore[arg-type]
        axes[1].plot(result["val_accuracy_history"], label=config.name)  # type: ignore[arg-type]

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def collect_prediction_examples(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_examples: int = 12,
) -> list[tuple[torch.Tensor, int, int]]:
    """从测试集中收集一组预测结果，用于画图。"""
    model.eval()
    collected: list[tuple[torch.Tensor, int, int]] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            for image, label, prediction in zip(images, labels, predictions, strict=True):
                collected.append((image.cpu(), int(label.item()), int(prediction.item())))
                if len(collected) >= num_examples:
                    return collected

    return collected


def plot_prediction_grid(
    examples: list[tuple[torch.Tensor, int, int]],
    output_path: Path,
) -> None:
    """把模型预测结果画成图，帮助直观看分类效果。"""
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for ax, (image, label, prediction) in zip(axes.ravel(), examples, strict=True):
        ax.imshow(unnormalize_image(image), cmap="gray")
        color = "green" if label == prediction else "crimson"
        ax.set_title(f"true={label}, pred={prediction}", color=color, fontsize=10)
        ax.axis("off")

    plt.suptitle("Test Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    """运行完整的第 4 周实验。"""
    ensure_directories()
    set_seed()
    device = get_device()

    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset = load_mnist_datasets()
    print(
        f"Dataset sizes: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    sample_grid_path = OUTPUT_DIR / "mnist_sample_grid.png"
    plot_sample_grid(train_dataset, sample_grid_path)

    experiments = [
        ExperimentConfig(
            name="baseline",
            hidden_dims=(256, 128),
            learning_rate=1e-3,
            batch_size=256,
            epochs=6,
            dropout=0.0,
            weight_decay=0.0,
        ),
        ExperimentConfig(
            name="regularized",
            hidden_dims=(256, 128),
            learning_rate=1e-3,
            batch_size=256,
            epochs=6,
            dropout=0.2,
            weight_decay=1e-4,
        ),
    ]

    results = [
        train_experiment(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
        )
        for config in experiments
    ]

    curves_path = OUTPUT_DIR / "pytorch_mlp_training_curves.png"
    plot_training_curves(results, curves_path)

    best_result = max(results, key=lambda item: float(item["test_accuracy"]))  # type: ignore[arg-type]
    best_config = best_result["config"]
    assert isinstance(best_config, ExperimentConfig)

    _, _, test_loader = make_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=256,
    )
    examples = collect_prediction_examples(
        model=best_result["model"],  # type: ignore[arg-type]
        data_loader=test_loader,
        device=device,
    )
    prediction_path = OUTPUT_DIR / "pytorch_mlp_prediction_grid.png"
    plot_prediction_grid(examples, prediction_path)

    print("\nTraining finished.")
    print(f"Saved sample grid: {sample_grid_path}")
    print(f"Saved training curves: {curves_path}")
    print(f"Saved prediction grid: {prediction_path}")
    print(f"Best experiment: {best_config.name}")


if __name__ == "__main__":
    main()
