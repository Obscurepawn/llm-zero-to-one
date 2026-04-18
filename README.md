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

notes/
  transformer-study-path.md
```

## 当前内容

- 一篇从线性回归、逻辑回归、损失函数和链式法则出发，逐步铺向 Transformer 的技术文章
- 一组用 NumPy 实现的线性回归与逻辑回归实验代码
- 梯度推导和链式法则计算图

## 本地运行

在实验目录下执行：

```bash
cd experiments/linear-and-logistic-regression
python linear_regression_numpy.py
python logistic_regression_numpy.py
python run_all.py
```
