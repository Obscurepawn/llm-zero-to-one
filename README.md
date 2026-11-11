# llm-zero-to-one

`llm-zero-to-one` 是一个从基础概念出发，逐步走向 Transformer 和大语言模型的学习型仓库。

当前仓库包含三类内容：

- `docs/`
  - 面向发布的技术文章和配套图片资源
- `experiments/`
  - 文中对应的可运行实验代码
- `notes/`
  - 额外的学习路径和补充笔记

## 当前结构

```text
docs/
  articles/
    from-linear-regression-to-transformer.md
    assets/

experiments/
  linear-and-logistic-regression/
  mlp-and-backpropagation/
  pytorch-mlp-mnist/
  text-token-embedding-language-model/

notes/
  transformer-study-path.md
```

## 当前内容

- [从线性回归到 Transformer 的技术文章](docs/articles/from-linear-regression-to-transformer.md)：串起线性回归、逻辑回归、损失函数和链式法则。
- [线性回归与逻辑回归](experiments/linear-and-logistic-regression/README.md)：NumPy 实现、梯度推导和训练曲线。
- [MLP 与反向传播](experiments/mlp-and-backpropagation/README.md)：对比单层感知机与两层 MLP，手写前向传播和梯度。
- [PyTorch MLP 与 MNIST](experiments/pytorch-mlp-mnist/README.md)：数据加载、训练、验证，以及 dropout 和 weight decay。
- [文本、Token、Embedding 和语言模型](experiments/text-token-embedding-language-model/README.md)：分词算法、next-token 样本和顺序信息实验。
- [Transformer 学习路径](notes/transformer-study-path.md)：后续学习安排。已实现的实验以 `experiments/` 为准。

## 本地运行

先运行只依赖 NumPy 和 Matplotlib 的回归实验。在仓库根目录执行以下命令（Python 3，POSIX shell）：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy matplotlib
cd experiments/linear-and-logistic-regression
python run_all.py
```

程序依次训练线性回归和逻辑回归，打印损失及模型参数，并在该实验的 `outputs/` 目录生成 5 张图，包括 `linear_regression_fit.png` 和 `logistic_regression_boundary.png`。也可以分别运行 `python linear_regression_numpy.py` 和 `python logistic_regression_numpy.py`。

后续实验的依赖不同：MLP 手写实验使用 NumPy 和 Matplotlib，MNIST 实验还使用 PyTorch 和 torchvision，Token/Embedding 实验使用 PyTorch、NumPy 和 Matplotlib。各实验的入口与输出见上面的目录链接。
