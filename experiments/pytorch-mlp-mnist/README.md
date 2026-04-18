# 第 4 周：用 PyTorch 重做 MLP，理解训练工程细节

这个目录对应学习计划中的第 4 周。第 2 周和第 3 周你已经分别手写过：

- 线性回归 / 逻辑回归
- 单层感知机 / 两层 MLP

到这一周，重点不再是自己推导每一层梯度，而是学会用深度学习框架把“数据加载、前向传播、loss、反向传播、优化器更新”串成一个标准训练流程。

## 目录结构

- `pytorch_mlp_mnist.py`
  - 用 `PyTorch` 实现一个用于 `MNIST` 分类的两层 MLP
  - 包含 `Dataset`、`DataLoader`、`nn.Module`、`optimizer`、`autograd`
  - 会自动保存训练曲线、样本图和预测结果图
- `run_all.py`
  - 第 4 周实验统一入口
- `pytorch_mlp_training_pipeline.svg`
  - 说明 `Dataset -> DataLoader -> MLP -> Loss -> backward -> optimizer.step()` 的训练骨架
- `outputs/`
  - 运行脚本后自动生成的图片
- `data/`
  - 默认用于存放下载的 `MNIST` 数据

## 这一周到底在学什么

如果第 3 周是在理解“一个 MLP 是怎么手写出来的”，那么第 4 周是在理解：

- 为什么训练代码几乎都会写成一个固定骨架
- `PyTorch` 到底帮你自动做了什么
- 为什么 `train loss` 下降了，不等于模型一定能泛化
- `dropout` 和 `weight decay` 为什么能帮助缓解过拟合

## 这份代码和前面几周怎么对齐

- 第 2 周你看到的是：
  - 自己写 `predict`
  - 自己写 `loss`
  - 自己写梯度
  - 自己更新参数
- 第 3 周你看到的是：
  - 自己写两层 MLP
  - 自己把误差一层层传回去
- 第 4 周你会看到：
  - `nn.Module` 帮你组织模型参数
  - `loss.backward()` 帮你自动计算梯度
  - `optimizer.step()` 帮你统一更新参数

所以这一周并不是“跳过原理”，而是把前面手写过的东西交给框架来做。

## 为什么第 3 周手写 MLP 没有 dropout，而这一周加了 dropout

这不是因为“用了 PyTorch 就必须加 dropout”，而是因为两周的教学重点不同。

第 3 周的重点是：

- 看清楚 MLP 的结构
- 手写前向传播和反向传播
- 理解非线性为什么重要

所以那一周的代码会故意只保留最核心的主干，不额外加入 `dropout`。这样做的目的是减少干扰，让你先把：

- 线性层
- 激活函数
- loss
- 链式法则
- backward

这条主线真正看清楚。

而第 4 周的重点已经变成了：

- 标准训练流程
- 训练集 / 验证集 / 测试集
- 泛化
- 过拟合
- 正则化

所以这一周才把 `dropout` 和 `weight decay` 加进来，并且故意同时保留：

- `baseline`：不加正则化
- `regularized`：加 `dropout` 和 `weight decay`

这样你就能直观看到：“模型会训练”和“模型能泛化”不是一回事。

## dropout 到底是什么

`dropout` 可以理解成一句话：

> 训练时，随机把一部分隐藏层神经元临时关掉。

如果某一层的激活写成 `a`，那么训练时可以粗略理解成：

```text
mask ~ Bernoulli(1 - p)
a_drop = a * mask / (1 - p)
```

这里：

- `p` 是丢弃概率
- `mask` 是一组 0/1 开关
- 被乘成 0 的神经元，这一轮就不参与后续计算

在这份代码里：

```python
nn.Dropout(0.2)
```

意思就是训练时随机丢掉大约 `20%` 的激活。

要注意一件事：

- 训练时，`dropout` 打开
- 推理时，`dropout` 关闭

也就是说，测试和实际使用模型时，不会继续随机屏蔽神经元。

## dropout 背后的想法是什么

它背后的核心思路是：

> 不要让网络过度依赖某几个固定神经元。

如果没有 `dropout`，网络可能会学成这样：

- 某个神经元特别关键
- 其他神经元总是在依赖它
- 模型把训练集里的局部模式记得很死

这就容易导致过拟合。

`dropout` 通过“训练时随机让一些神经元缺席”，逼着网络学得更稳一些：

- 不能总依赖同一个神经元
- 不同路径都得能工作
- 学到的表示会更分散，也更鲁棒

你也可以把它理解成一种很便宜的“模型集成”思路：

- 每一次随机丢弃神经元，都像是在训练一个略微不同的子网络
- 最后推理时使用完整网络，效果有点像把很多子网络的能力合在一起

## 运行前提

这份代码需要：

- `torch`
- `torchvision`
- `matplotlib`

如果本机还没有安装，可以先安装：

```bash
python -m pip install torch torchvision matplotlib
```

## 如何运行

在当前目录执行：

```bash
python pytorch_mlp_mnist.py
```

如果你想使用统一入口：

```bash
python run_all.py
```

脚本默认会：

1. 下载或读取 `MNIST`
2. 切分出训练集、验证集、测试集
3. 训练两个版本的模型
   - `baseline`: 不加正则化
   - `regularized`: 加 `dropout` 和 `weight decay`
4. 输出训练日志
5. 保存图片到 `outputs/`

## 运行后建议重点观察什么

### 1. `Dataset` 和 `DataLoader` 在做什么

- `Dataset` 负责“我这里有数据，单个样本怎么取”
- `DataLoader` 负责“每次取多大一批，是否打乱，如何迭代”

这两者是你以后读几乎所有 `PyTorch` 训练代码时都会见到的骨架。

### 2. `nn.Module` 和手写 MLP 的关系

你可以把这里的 `MLPClassifier` 和第 3 周的 `TwoLayerMLP` 对照着看：

- 本质上仍然是若干层线性层加激活函数
- 差别只是参数注册、前向调用和反向传播交给了框架

### 3. `loss.backward()` 到底在替代什么

它替代的正是你第 3 周在 `backward()` 里手写的那一整套链式法则。

只是这次你不需要自己一层层写：

- `dZ2`
- `dW2`
- `dA1`
- `dZ1`
- `dW1`

框架会根据计算图自动算出来。

### 4. 为什么要同时保留训练集、验证集、测试集

- 训练集：用来更新参数
- 验证集：用来观察泛化趋势和调超参数
- 测试集：只在最后看一次真正效果

这一点非常重要。模型会训练，不代表模型一定能泛化。

### 5. 为什么要同时训练 `baseline` 和 `regularized`

这一步是为了让你对以下概念建立直觉：

- 不加正则化的模型，可能在训练集上越来越好，但验证集未必同步提升
- `dropout`
- `weight decay`

这两个东西都不是“让训练更快”，而是帮助模型不过度记住训练集细节。

## 默认输出图片

脚本默认会生成这些图：

- `outputs/mnist_sample_grid.png`
  - 随机抽取的 `MNIST` 样本，帮助你确认数据长什么样
- `outputs/pytorch_mlp_training_curves.png`
  - 两个模型的训练 loss 和验证准确率曲线
- `outputs/pytorch_mlp_prediction_grid.png`
  - 测试集上的部分预测结果

## 这周学完后，你至少应该能回答的问题

1. `Dataset` 和 `DataLoader` 分别解决什么问题？
2. `nn.Module` 和第 3 周手写的 MLP 本质上有什么对应关系？
3. `loss.backward()` 实际上替代了你之前手写的哪一部分？
4. 为什么训练集、验证集、测试集不能混着用？
5. 为什么“训练 loss 很低”不等于“模型泛化很好”？
