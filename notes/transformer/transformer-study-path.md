# 从 0 到 Transformer：12 周学习路线

这份路线适合“几乎 0 基础，但希望真正理解 Transformer 原理”的学习者。目标不是先会调库，而是先建立一条稳定的理解链路：

`数学和张量直觉 -> MLP -> 反向传播与训练 -> 序列建模 -> 注意力 -> Transformer`

## 使用方式

- 建议投入：每周 10 到 14 小时。
- 推荐节奏：工作日每天 1.5 到 2 小时，周末半天复盘和做实验。
- 每周都要有代码产出，不能只看视频或文章。
- 每周结束时，必须用自己的话解释关键概念；说不清就说明还没真正理解。

## 学习原则

1. 先理解输入、参数、输出和损失之间的关系，再记公式。
2. 每学一个模块，就做一个最小可运行实验。
3. 不要一开始追大模型、微调、RLHF、分布式训练。
4. 优先建立“为什么这样设计”的直觉，再补数学细节。

## 环境准备

- Python 3
- Jupyter Notebook 或 VS Code
- `numpy`
- `matplotlib`
- `pytorch`
- `torchvision`
- `jupyter`

可选安装命令：

```bash
pip install numpy matplotlib torch torchvision jupyter
```

---

## 第 1 周：Python、Numpy、张量和最小数学直觉

### 本周目标

建立后续学习所需的最小工具基础，能看懂张量运算，理解向量、矩阵、维度和矩阵乘法。

### 必学概念

- Python 基础语法、函数、类、列表、字典
- `numpy.ndarray`
- 标量、向量、矩阵、张量
- shape、broadcasting
- 矩阵乘法和点积
- 导数、偏导、链式法则的直观含义

### 实践任务

- 用 `numpy` 完成向量加法、矩阵乘法、转置、切片。
- 自己写一个函数，输入矩阵 `X` 和权重 `W`，输出 `XW + b`。
- 画出一元函数和导数的关系，例如 `y = x^2`。

### 过关标准

- 你能解释为什么神经网络里大量操作都能写成矩阵乘法。
- 你能看懂一个 shape 变化过程，比如 `(32, 128) x (128, 64) -> (32, 64)`。

### 本周输出

- 一个 `numpy` 练习 notebook
- 一页自己的学习笔记：什么是张量、为什么矩阵乘法重要

---

## 第 2 周：线性回归、逻辑回归和损失函数

### 本周目标

理解“模型是什么、参数怎么更新、loss 为什么能指导学习”。

### 必学概念

- 监督学习
- 特征、标签、样本
- 线性回归
- 逻辑回归
- 均方误差 MSE
- 交叉熵 Cross Entropy
- 梯度下降的基本思想

### 实践任务

- 用 `numpy` 手写线性回归，拟合一组简单数据。
- 用 `numpy` 实现二分类逻辑回归。
- 观察学习率过大、过小时 loss 的变化。

### 过关标准

- 你能解释模型、参数、预测值、loss 四者之间的关系。
- 你能理解“训练”本质上是不断更新参数，让 loss 下降。

### 本周输出

- 一个手写线性回归 notebook
- 一个手写逻辑回归 notebook

---

## 第 3 周：多层感知机 MLP 和反向传播

### 本周目标

从单层模型进入真正的神经网络，搞懂为什么加非线性后表达能力增强，以及反向传播在做什么。

### 必学概念

- 感知机
- 全连接层 Linear Layer
- 激活函数：Sigmoid、Tanh、ReLU
- 前向传播
- 反向传播
- 计算图
- 链式法则在网络中的应用

### 实践任务

- 用 `numpy` 手写一个两层 MLP。
- 在一个简单二维分类数据集上训练它。
- 可视化决策边界，比较“线性模型”和“MLP”的差异。

### 过关标准

- 你能解释为什么没有激活函数时，多层线性层仍然等价于一层线性层。
- 你能用自己的话描述反向传播：误差如何一层层传回去。

### 本周输出

- 手写两层 MLP
- 一份反向传播的流程图或文字解释

---

## 第 4 周：用 PyTorch 重做 MLP，理解训练工程细节

### 本周目标

学会使用深度学习框架，同时建立对训练过程的直觉。

### 必学概念

- `torch.Tensor`
- `Dataset` 和 `DataLoader`
- `nn.Module`
- `optimizer`
- `autograd`
- batch、epoch、learning rate
- 训练集、验证集、测试集
- 过拟合
- dropout、weight decay 的直觉

### 实践任务

- 用 PyTorch 实现一个 MLP，跑 MNIST 手写数字分类。
- 记录训练 loss、验证准确率。
- 试着调整学习率、batch size、隐藏层大小。

### 过关标准

- 你能读懂一个标准 PyTorch 训练循环。
- 你能解释“模型会训练”和“模型能泛化”不是一回事。

### 本周输出

- 一个 MNIST 分类脚本或 notebook
- 一张训练曲线图

---

## 第 5 周：文本、Token、Embedding 和语言模型任务

### 本周目标

从“固定长度特征”切换到“序列数据”视角，理解文本是如何进入神经网络的。

### 必学概念

- 什么是序列数据
- token、vocabulary
- one-hot 编码
- embedding 的作用
- 语言模型任务：根据前文预测下一个 token
- 位置顺序为什么重要

### 实践任务

- 做一个字符级别的小数据集，例如一段中文或英文文本。
- 自己写 tokenizer 的最小版本。
- 对比 one-hot 和 embedding 的维度与表达方式。
- 用 embedding 层把 token id 映射成向量。

### 过关标准

- 你能解释为什么文本不能直接拿字符串送入神经网络。
- 你能说明 embedding 比 one-hot 更适合学习语义表示。

### 本周输出

- 一个最小 tokenizer
- 一个 embedding 演示 notebook

---

## 第 6 周：RNN 和序列建模的第一步

### 本周目标

理解经典序列模型如何逐步处理输入，以及它的局限是什么。

### 必学概念

- 循环神经网络 RNN
- 隐状态 hidden state
- 时间步 time step
- 参数共享
- 展开后的计算图
- BPTT：Backpropagation Through Time
- 梯度消失和梯度爆炸的直觉

### 实践任务

- 用 PyTorch 做一个简单字符级 RNN，预测下一个字符。
- 可视化不同时间步的输入和隐藏状态变化。
- 观察序列变长后训练变难的现象。

### 过关标准

- 你能解释 RNN 为什么适合处理变长序列。
- 你能解释 RNN 为什么难以处理很长距离的依赖。

### 本周输出

- 一个字符级 RNN 训练脚本
- 一页“RNN 的优势和问题”总结

---

## 第 7 周：LSTM、GRU 和 seq2seq 背景

### 本周目标

理解在 Transformer 之前，人们如何改进序列模型，以及这些方法仍然存在哪些瓶颈。

### 必学概念

- LSTM 的门控思想
- GRU 的基本结构
- encoder-decoder
- seq2seq
- teacher forcing
- 长距离依赖问题
- 串行计算导致难并行

### 实践任务

- 阅读并画出 LSTM/GRU 的结构示意图。
- 跑一个简单 seq2seq 例子，例如数字序列翻转或复制任务。
- 对比 RNN 和 LSTM 在较长序列上的表现。

### 过关标准

- 你能解释 LSTM 相比普通 RNN 改进了什么。
- 你能说明为什么 seq2seq 仍然会在长序列上遇到瓶颈。

### 本周输出

- 一个 seq2seq 小实验
- 一张“RNN -> LSTM/GRU -> seq2seq”的发展脉络图

---

## 第 8 周：注意力机制 Attention

### 本周目标

正式进入 Transformer 的核心前置知识，建立“模型可以有选择地关注重要位置”的直觉。

### 必学概念

- 注意力的动机
- query、key、value 的角色
- 相似度打分
- softmax 权重
- 加权求和
- attention 解决了什么问题

### 实践任务

- 用一个简单例子手算 attention 权重。
- 用 `numpy` 实现最小 attention。
- 在 seq2seq 背景下理解 decoder 为什么需要“关注”不同输入位置。

### 过关标准

- 你能不用公式，直接用自然语言解释 Q、K、V 各自代表什么。
- 你能解释 attention 为什么能缓解“所有信息都挤进一个固定向量”的问题。

### 本周输出

- 一个手写 attention notebook
- 一页 Q/K/V 解释笔记

---

## 第 9 周：Self-Attention、多头机制和位置编码

### 本周目标

理解 Transformer block 中最关键的几个部件，并弄清它们为什么要同时出现。

### 必学概念

- self-attention
- scaled dot-product attention
- multi-head attention
- positional encoding
- mask 的作用
- 残差连接 residual connection
- layer normalization

### 实践任务

- 用 `numpy` 或 PyTorch 实现一个最小 self-attention。
- 输入一个长度为 `N` 的序列，观察 attention score 矩阵形状。
- 画图理解不同 head 学到不同关注模式的直觉。
- 比较“有位置编码”和“无位置编码”的区别。

### 过关标准

- 你能解释 self-attention 与普通 attention 的区别。
- 你能解释为什么 Transformer 需要位置编码。
- 你能说明多头注意力不是简单重复，而是在不同子空间里学习关系。

### 本周输出

- 一个 self-attention 最小实现
- 一张 Transformer block 草图

---

## 第 10 周：Transformer Encoder

### 本周目标

把前面所有模块拼起来，理解 Encoder 是如何一层层编码输入序列的。

### 必学概念

- 输入 embedding
- 位置编码相加
- multi-head self-attention
- feed-forward network
- add & norm
- 堆叠多层 encoder

### 实践任务

- 用 PyTorch 实现一个最小 Transformer Encoder block。
- 输入一段 token 序列，打印每层输出 shape。
- 结合图示解释一个 token 的表示如何被上下文更新。

### 过关标准

- 你能从输入到输出完整解释一个 encoder block 的数据流。
- 你能解释 feed-forward network 在每个位置上做了什么。

### 本周输出

- 一个最小 encoder block 实现
- 一页“encoder block 拆解说明”

---

## 第 11 周：Transformer Decoder 和完整架构

### 本周目标

理解完整 Transformer 的编码器-解码器结构，并区分训练时与推理时的差别。

### 必学概念

- masked self-attention
- encoder-decoder attention
- 自回归生成
- 训练阶段与推理阶段的输入差异
- next-token prediction
- causal mask

### 实践任务

- 画出完整 Transformer 架构图。
- 实现最小 decoder block，尤其是 causal mask。
- 用一个极小数据集跑通 next-token prediction。

### 过关标准

- 你能解释 decoder 为什么不能看到未来 token。
- 你能区分 encoder-only、decoder-only、encoder-decoder 三类模型的基本用途。

### 本周输出

- 一个最小 decoder block
- 一张完整 Transformer 结构图

---

## 第 12 周：从原理到小复现

### 本周目标

把前 11 周内容串成一个完整闭环，做到“能讲清 + 能写出一个最小版本 + 能定位关键设计”。

### 必学概念

- 整体复盘
- 数据流和 shape 变化
- 参数量的基本估算
- 训练不稳定时的常见原因
- Transformer 相比 RNN 的优点和代价

### 实践任务

- 用 PyTorch 写一个迷你版 Transformer。
- 在一个很小的字符级或词级数据集上训练 next-token prediction。
- 记录 loss 曲线和几个生成结果。
- 写一份 1 到 2 页总结，回答下面 5 个问题：

1. 为什么 MLP 不能直接替代序列模型？
2. 为什么 RNN 难并行？
3. Attention 在解决什么核心问题？
4. Transformer 为什么需要位置编码？
5. Multi-head attention 为什么有效？

### 过关标准

- 你能独立画出 Transformer block，并解释每个模块在做什么。
- 你能大致读懂一份常见的 PyTorch Transformer 实现。
- 你能把“从 MLP 到 Transformer”的发展逻辑讲顺。

### 本周输出

- 一个迷你 Transformer 项目
- 一份最终复盘文档

---

## 每周固定节奏建议

### 周一到周三

- 学概念
- 做最小笔记
- 跑最小示例

### 周四到周五

- 自己动手实现核心代码
- 看 loss、shape、输出结果

### 周末

- 复盘本周
- 写总结
- 补掉没搞懂的点

---

## 推荐的学习顺序，不建议跳步

1. 先把 MLP 和反向传播搞懂。
2. 再理解序列为什么麻烦。
3. 再学 RNN/LSTM，理解旧方案的限制。
4. 再进入 Attention 和 Transformer。

如果直接跳到 Transformer，大概率会变成“记住模块名字”，但不知道这些设计为什么存在。

---

## 你每一阶段最少要能回答的问题

### MLP 阶段

- 为什么需要激活函数？
- 反向传播到底在算什么？

### 序列模型阶段

- 为什么固定长度输入的方法不适合文本？
- RNN 为什么会忘掉很远的信息？

### Attention 阶段

- Q、K、V 分别是什么？
- 为什么注意力权重可以理解成“相关性”？

### Transformer 阶段

- 为什么它可以并行？
- 为什么它还需要位置编码？
- 为什么一个 block 里既有 attention 又有前馈层？

---

## 常见误区

- 只看视频，不写代码。
- 只背公式，不理解数据流。
- 过早追求大模型训练和工程技巧。
- 学了 Transformer，但说不清 RNN 的问题。

---

## 12 周结束后，你应该达到的水平

- 能看懂主流入门级 Transformer 教程和代码实现。
- 能读懂 `embedding -> attention -> ffn -> residual -> norm` 这条主线。
- 能手写一个迷你版 Transformer 或至少复现一个教学实现。
- 能继续往 BERT、GPT、ViT、微调和大模型工程方向深入。

## 下一步建议

12 周结束后，建议继续选一个方向深入：

- 方向 1：BERT / GPT 架构差异
- 方向 2：训练技巧，比如 warmup、scheduler、label smoothing
- 方向 3：NLP 实战，比如文本分类、摘要、问答
- 方向 4：大模型推理和微调

如果你执行这份计划时发现负担过重，可以把 12 周拉长到 16 到 20 周，但不建议压缩到 6 周以内。
