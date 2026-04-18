# 从 0 到 Transformer 与大语言模型：16 周教学计划

下面这份计划是给 0 基础学习者准备的。安排思路很明确：先把数学、线性模型、反向传播和序列建模这些基础打牢，再进入注意力机制和 Transformer，最后补上大语言模型阶段最常见的训练、推理和应用问题。

整条线索如下：

`数学和张量直觉 -> MLP -> 反向传播与训练 -> 序列建模 -> 注意力 -> Transformer -> 大语言模型`

前 12 周是基础部分，目标是把 Transformer 的原理真正讲顺；后 4 周是进阶部分，目标是把视野从 Transformer 进一步延伸到大语言模型。

## 教学说明

- 建议每周投入 10 到 14 小时，尽量保持固定节奏。
- 工作日以学习和小实验为主，周末安排复盘、整理笔记和完成作业。
- 每周都要有代码产出，不能只停留在看视频或看文章。
- 学完一个模块后，要能用自己的话讲清楚关键概念；讲不清楚，就说明还需要回头补。

## 学习要求

1. 先理解输入、参数、输出和损失之间的关系，再记公式。
2. 每学一个模块，就做一个最小可运行实验。
3. 前 12 周先不追求大模型训练、对齐和分布式工程，先把原理学扎实。
4. 优先建立“为什么这样设计”的直觉，再补数学细节。

## 环境准备

- Python 3
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

### 代码实践

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

### 代码实践

- 对应仓库目录：`experiments/linear-and-logistic-regression/`
- 先读 `linear_regression_numpy.py`，把前向计算、MSE、梯度和参数更新串起来
- 再读 `logistic_regression_numpy.py`，理解 `sigmoid`、交叉熵和分类边界
- 最后运行 `run_all.py`，对照生成的 `outputs/` 图像观察学习率和边界变化

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
- 结合一张结构图，说明为什么线性回归和逻辑回归都可以看成最简单的单层感知机。

### 过关标准

- 你能解释为什么没有激活函数时，多层线性层仍然等价于一层线性层。
- 你能用自己的话描述反向传播：误差如何一层层传回去。

### 代码实践

- 对应仓库目录：`experiments/mlp-and-backpropagation/`
- 阅读 `single_layer_models_as_perceptrons.svg`，先建立“线性回归 / 逻辑回归 / 两层 MLP”三者的结构关系
- 阅读 `mlp_numpy.py` 中的 `SingleLayerPerceptron`，把它和第 2 周的逻辑回归代码对照起来看
- 阅读 `TwoLayerMLP` 和 `backward()`，重点看误差如何从输出层传回隐藏层
- 运行 `run_all.py`，对照 `outputs/mlp_vs_single_layer_loss.png` 和 `outputs/mlp_vs_single_layer_boundary.png`

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

### 代码实践

- 对应仓库目录：`experiments/pytorch-mlp-mnist/`
- 先看 `pytorch_mlp_training_pipeline.svg`，把 `Dataset -> DataLoader -> model -> loss -> backward -> optimizer.step()` 这条主线理顺
- 阅读 `pytorch_mlp_mnist.py`，重点对照第 3 周代码看哪些部分交给了 `nn.Module`、`autograd` 和 `optimizer`
- 运行 `run_all.py`，观察 `outputs/mnist_sample_grid.png`、`outputs/pytorch_mlp_training_curves.png` 和 `outputs/pytorch_mlp_prediction_grid.png`

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

### 代码实践

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

### 代码实践

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

### 代码实践

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

### 代码实践

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

### 代码实践

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

### 代码实践

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

### 代码实践

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

### 代码实践

- 一个迷你 Transformer 项目
- 一份最终复盘文档

---

## 第 13 周：从 Transformer 到预训练语言模型

### 本周目标

把“Transformer 是一种结构”进一步过渡到“预训练语言模型是一类训练出来的模型”，建立对 BERT、GPT、T5 这类模型的基本认识。

### 必学概念

- 预训练 pretraining
- 语言模型目标
- causal language model
- masked language model
- decoder-only、encoder-only、encoder-decoder
- tokenizer 与子词切分
- 参数规模、上下文长度、语料规模的直观关系

### 实践任务

- 画出 BERT、GPT、T5 三类模型的结构对比图。
- 用一个开源 tokenizer 对同一段文本做切分，观察 token 化结果。
- 跑一个小型预训练模型，查看 next-token prediction 或 masked token prediction 的输出。

### 过关标准

- 你能解释为什么 GPT 更适合生成任务，BERT 更常用于理解任务。
- 你能说明“Transformer”是结构，“预训练语言模型”是训练结果，两者不是同一个概念。

### 代码实践

- 一份 BERT / GPT / T5 对比笔记
- 一个 tokenizer 与预训练模型的最小实验

---

## 第 14 周：指令微调与对话模型

### 本周目标

理解“会续写文本的基础模型”是如何进一步变成“能按要求回答问题的对话模型”的。

### 必学概念

- base model 与 instruct model
- 指令微调 instruction tuning
- supervised fine-tuning
- chat template
- system / user / assistant 角色
- LoRA / QLoRA 的基本直觉
- 对齐 alignment 的基本含义

### 实践任务

- 自己整理一个小型指令数据集，格式化成 input / output 对。
- 对比同一个基础模型在普通续写提示和指令提示下的输出差异。
- 阅读一个最小 LoRA 微调示例，重点看数据格式、损失目标和训练流程。

### 过关标准

- 你能解释为什么“会续写”并不等于“会按要求回答”。
- 你能说明指令微调主要改变的是模型的输出风格和任务适配能力。

### 代码实践

- 一份小型指令数据样例
- 一页 base model 与 instruct model 的差异总结

---

## 第 15 周：大语言模型推理、采样与 RAG

### 本周目标

理解模型部署和实际使用时最常见的几个问题：为什么同一个模型会有不同回答，为什么需要检索增强，为什么上下文长度会成为限制。

### 必学概念

- greedy decoding
- temperature
- top-k / top-p
- repetition penalty
- KV cache
- context window
- embedding 检索
- RAG 的基本流程

### 实践任务

- 用同一个小模型比较 greedy、temperature、top-p 等不同解码方式的输出。
- 观察上下文变长后推理速度和显存占用的变化。
- 做一个最小 RAG 演示，例如准备 5 到 10 篇短文档并完成检索问答。

### 过关标准

- 你能解释为什么同一个模型在不同采样设置下会给出不同回答。
- 你能说明 RAG 解决了什么问题，又不能解决什么问题。

### 代码实践

- 一份解码策略对比实验记录
- 一个最小 RAG demo

---

## 第 16 周：评估、安全与完整项目视角

### 本周目标

站在更完整的项目视角，理解大语言模型不仅要“能生成”，还要“可评估、可比较、可控、可落地”。

### 必学概念

- eval set 与人工评估
- hallucination
- prompt injection
- jailbreak 的基本风险
- latency 与吞吐
- 量化 quantization 的基本作用
- prompt、微调、RAG 三种方案的取舍

### 实践任务

- 为一个简单任务设计一套小型评测集，例如问答或摘要。
- 比较 prompt-only、RAG、指令模型三种方案在同一任务上的表现差异。
- 写一份最终总结，说明如果从零做一个小型 LLM 应用，你会如何选择方案。

### 过关标准

- 你能说清楚一个 LLM 项目为什么必须重视评估，而不能只看几条“感觉不错”的输出。
- 你能根据任务特点初步判断应该优先尝试 prompt、RAG 还是微调。

### 代码实践

- 一份小型评测报告
- 一份“从 Transformer 到 LLM 应用”的最终总结

---

## 建议的周内安排

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

## 建议按这个顺序推进

1. 先把线性模型、MLP 和反向传播搞懂。
2. 再理解序列建模为什么会麻烦。
3. 再学 RNN/LSTM，理解旧方案的限制。
4. 再进入 Attention 和 Transformer。
5. 最后过渡到预训练语言模型、指令微调、推理和应用。

如果直接跳到 Transformer，往往会变成“记住模块名字”，但不知道这些设计为什么存在；如果又直接跳到大语言模型阶段，往往会只会调用接口，却说不清模型为什么这样训练、为什么这样生成。

---

## 你每一阶段最少要能回答的问题

### 线性模型阶段

- 模型、参数、预测值和损失函数分别是什么？
- 为什么最小化损失可以转化成参数更新问题？

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

### 大语言模型阶段

- 预训练模型和指令模型有什么区别？
- 为什么同一个模型在不同采样设置下会给出不同回答？
- RAG 在什么情况下有用，什么情况下不够用？

---

## 常见误区

- 只看视频，不写代码。
- 只背公式，不理解数据流。
- 还没搞清楚损失函数和梯度，就急着追大模型训练和工程技巧。
- 学了 Transformer，但说不清 RNN 的问题。
- 只会调用模型接口，却说不清预训练、指令微调和 RAG 的区别。

---

## 16 周结束后，你应该达到的水平

- 能看懂主流入门级 Transformer 教程和代码实现。
- 能读懂 `embedding -> attention -> ffn -> residual -> norm` 这条主线。
- 能手写一个迷你版 Transformer 或至少复现一个教学实现。
- 能分清 BERT、GPT、T5 这类模型的大体结构差异。
- 能理解指令微调、采样、RAG、评估这些大语言模型中的常见概念。
- 能继续往微调、推理优化、Agent、RAG 或多模态方向深入。

## 后续深入方向

16 周课程结束后，可以继续选一个方向往下走：

- 方向 1：预训练与微调，例如 SFT、LoRA、DPO
- 方向 2：推理优化，例如量化、KV cache、推理框架
- 方向 3：知识增强应用，例如 RAG、工具调用、Agent
- 方向 4：多模态方向，例如图文模型和视觉语言模型

如果学习过程中感到负担较重，可以把这份 16 周计划拉长到 20 到 24 周，但不建议压缩得太快。
