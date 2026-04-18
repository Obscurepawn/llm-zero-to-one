# 第 5 周：文本、Token、Embedding 和语言模型任务

这个目录对应学习计划中的第 5 周，重点是把“字符串是怎么进入神经网络的”这条主线真正跑通。

前四周你主要在处理固定长度的数值特征：

- 第 2 周：线性回归、逻辑回归
- 第 3 周：单层感知机、两层 MLP
- 第 4 周：PyTorch 训练流程

到了这一周，输入不再是已经整理好的数值向量，而是原始文本。这里最重要的不是先追求复杂模型，而是先弄清楚下面这条链路：

`raw text -> token -> token id -> embedding -> next-token prediction`

## 目录结构

- `tokenizer_and_embedding_demo.py`
  - 一个最小字符级 tokenizer
  - 把文本切成 token id，并构造 next-token 训练样本
  - 对比 one-hot 和 embedding lookup
  - 用一个顺序敏感的小实验说明为什么位置重要
- `tokenizer_training_algorithms_demo.py`
  - 用一个文件并排演示字符级、BPE、WordPiece、Unigram 这几类 tokenizer
  - 展示它们各自的“训练”方式到底在做什么
  - 输出每一步 merge / 概率更新后的摘要
- `order_models_architecture.svg`
  - 并排展示 `BagOfEmbeddingsLM` 和 `PositionAwareLM` 的前向流程
  - 标出输入形状、embedding 输出、池化/展平、logits 和 loss
- `order_models_flow_explanation.md`
  - 结合结构图解释两个模型为什么一个会丢掉顺序、一个能保留顺序
- `run_all.py`
  - 第 5 周统一入口
- `outputs/`
  - 运行脚本后自动生成的图和预览文本

## 这份代码具体在演示什么

### 1. 为什么文本不能直接送进神经网络

神经网络处理的是数值张量，不会直接理解字符串。

所以脚本先做了一个最小字符级 tokenizer，把文本：

```text
language models
```

变成一串整数 id。

你可以把这一步理解成：

- 先建立词表 `vocabulary`
- 再把每个 token 映射成一个整数

这就是“文本数值化”的第一步。

### 2. 语言模型任务到底是什么

脚本里用了滑动窗口构造最基础的语言模型样本：

```text
[t0, t1, t2, t3, t4, t5] -> t6
```

也就是：

- 给模型一个前文 `context`
- 让模型预测下一个 token `target`

这就是大语言模型最核心的预训练任务雏形。

### 3. one-hot 和 embedding 到底差在哪

脚本会把同一串 token 同时表示成：

- one-hot 向量
- embedding 向量

并生成 `outputs/one_hot_vs_embedding_comparison.png`。

你运行后可以重点看三件事：

1. one-hot 的维度直接等于词表大小
2. embedding 可以把 token 压到更低维的稠密向量里
3. `one-hot @ embedding_weight` 和 `nn.Embedding` lookup 数值上是等价的

这一点非常重要，因为它说明：

- embedding 不是神秘的新对象
- 它本质上就是“按 token id 去查一个可训练的向量表”

### 4. 为什么顺序重要

这一周最容易被忽略的一点是：

> 文本不是“无序特征集合”，而是“有顺序的序列”。

为了把这件事讲得足够直观，脚本做了一个故意设计过的小实验：

- `['a', 'b'] -> 'x'`
- `['b', 'a'] -> 'y'`

然后比较两个模型：

- `bag_of_embeddings`
  - 把上下文 embedding 做平均
  - 会丢掉顺序信息
- `position_aware`
  - 保留每个位置的 embedding
  - 能区分 `ab` 和 `ba`

运行后会生成 `outputs/position_order_importance.png`。

你应该能看到：

- 忽略顺序的模型准确率会上不去
- 保留顺序的模型可以轻松学会任务

这正是后面学习 RNN、位置编码和 Transformer 时必须建立的直觉基础。

## 如何运行

在当前目录执行：

```bash
python tokenizer_and_embedding_demo.py
```

如果你想用统一入口：

```bash
python run_all.py
```

如果你想单独看 tokenizer 训练算法 demo：

```bash
python tokenizer_training_algorithms_demo.py
```

## 默认输出

- `outputs/tokenizer_dataset_preview.txt`
  - 词表、token 频率、编码示例、next-token 样本预览
- `outputs/one_hot_vs_embedding_comparison.png`
  - one-hot 和 embedding 的表示方式对比
- `outputs/position_order_importance.png`
  - 忽略顺序 vs 保留顺序的训练结果对比
- `outputs/tokenizer_training_algorithms_demo.txt`
  - 字符级、BPE、WordPiece、Unigram 四种 tokenizer 的训练摘要与分词示例

## 建议阅读顺序

1. 先看 `CharTokenizer`
2. 再看 `build_next_token_dataset()`
3. 接着看 `plot_one_hot_vs_embedding()`
4. 最后看 `BagOfEmbeddingsLM` 和 `PositionAwareLM`

## 这周学完后，你至少应该能回答的问题

1. 为什么字符串不能直接送进神经网络？
2. token、vocabulary、token id 三者是什么关系？
3. 为什么 embedding 比 one-hot 更适合做可学习表示？
4. “根据前文预测下一个 token” 这个任务本质上在做什么？
5. 为什么序列模型不能忽略位置顺序？
