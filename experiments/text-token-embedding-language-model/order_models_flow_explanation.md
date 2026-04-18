# BagOfEmbeddingsLM 与 PositionAwareLM 全流程说明

配套结构图：
- [order_models_architecture.svg](/home/obscure/LLM-from-zero-to-one/experiments/text-token-embedding-language-model/order_models_architecture.svg)

## 1. 两个模型的共同主线

两者都在做同一件事：

`context_ids -> embedding -> 某种上下文压缩/整理 -> linear -> logits -> cross entropy loss`

其中：
- `context_ids` 是输入 token id
- `embedding` 把离散 token id 变成连续向量
- `linear` 对词表中每个候选 token 打分
- `logits` 是还没经过 softmax 的分数
- `loss` 用真实 `target_ids` 监督训练

## 2. 输入张量长什么样

在这个顺序实验里：
- `context_size = 2`
- `vocab_size = 8`
- `embedding_dim = 8`

所以一个 batch 的输入大致是：

```text
context_ids.shape = [batch_size, 2]
```

例如：

```text
[
  [0, 1],   # ['a', 'b']
  [1, 0],   # ['b', 'a']
  [2, 3],   # ['c', 'd']
]
```

## 3. 经过 embedding 后会变成什么

embedding 层会把每个 token id 查成一个长度为 8 的向量：

```text
embeddings.shape = [batch_size, 2, 8]
```

可以理解成：
- 第 0 维：batch 里的第几个样本
- 第 1 维：上下文里的第几个位置
- 第 2 维：这个 token 向量的第几维

## 4. BagOfEmbeddingsLM 为什么会丢掉顺序

代码位置：
- [BagOfEmbeddingsLM](/home/obscure/LLM-from-zero-to-one/experiments/text-token-embedding-language-model/tokenizer_and_embedding_demo.py#L575)

它的关键一步是：

```python
pooled = embeddings.mean(dim=1)
```

这会把两个位置的向量直接求平均：

```text
[a_embed, b_embed] -> (a_embed + b_embed) / 2
[b_embed, a_embed] -> (b_embed + a_embed) / 2
```

两者结果完全一样，所以顺序信息丢了。

然后它再做：

```python
nn.Linear(embedding_dim, vocab_size)
```

也就是：
- 输入 `[batch_size, 8]`
- 输出 `[batch_size, 8]`

输出的 8 维就是对 8 个候选 token 的分数。

## 5. PositionAwareLM 为什么能保留顺序

代码位置：
- [PositionAwareLM](/home/obscure/LLM-from-zero-to-one/experiments/text-token-embedding-language-model/tokenizer_and_embedding_demo.py#L619)

它的关键一步是：

```python
flattened = embeddings.reshape(context_ids.size(0), -1)
```

对 `context_size = 2, embedding_dim = 8` 来说，这相当于：

```text
[position0_embed, position1_embed]
-> 把两个 8 维向量拼成一个 16 维向量
```

于是：

```text
concat(a, b) != concat(b, a)
```

所以顺序被保留了。

然后它再做：

```python
nn.Linear(context_size * embedding_dim, vocab_size)
```

也就是：
- 输入 `[batch_size, 16]`
- 输出 `[batch_size, 8]`

## 6. logits 和 loss 怎么理解

两个模型最后都会输出：

```text
logits.shape = [batch_size, vocab_size]
```

意思是：
- 对 batch 中每个样本
- 给词表里的每个 token 打一个分

然后用：

```python
CrossEntropyLoss(logits, target_ids)
```

把这些分数和真实标签比较。

## 7. 一句话总结差别

`BagOfEmbeddingsLM`：
- 把多个位置平均掉
- 只保留“有哪些 token”
- 不保留“token 的位置顺序”

`PositionAwareLM`：
- 把每个位置单独保留下来
- 让模型知道“第 0 个位置是谁，第 1 个位置是谁”
- 因此能区分顺序
