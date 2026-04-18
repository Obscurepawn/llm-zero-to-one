# 第 6 周实验设计：RNN 和序列建模的第一步

## 背景与目标

当前仓库已经覆盖了前 5 周的学习主线：

- 第 2 周：线性回归、逻辑回归和损失函数
- 第 3 周：多层感知机 MLP 和反向传播
- 第 4 周：用 PyTorch 重做 MLP，理解训练工程细节
- 第 5 周：文本、Token、Embedding 和语言模型任务

第 6 周需要承接第 5 周的结果，把“文本如何变成 token / embedding”继续推进到“模型如何按时间步处理序列”，并为第 7 周的 LSTM / GRU 做自然铺垫。

本周目标不是覆盖所有序列模型，而是聚焦最基础的 vanilla RNN，帮助初学者建立下面这条主线的直觉：

`token ids -> embeddings -> hidden state over time -> next-character prediction`

同时，实验还要让学习者直观看到 RNN 的两个核心特征：

1. 它可以处理变长序列，并且通过 hidden state 累积上下文。
2. 它在长距离依赖上会变得难训练，这正是后续需要 LSTM / GRU 的原因。

## 设计原则

- 对齐前五周目录风格：每周一个独立实验目录，包含 `README.md`、统一入口 `run_all.py`、脚本、图片输出和必要的补充文档。
- 保持教学导向：代码优先服务“看懂原理”，而不是追求最短实现或工程抽象。
- 尽量零额外依赖：使用仓库内置的短小英文语料，不新增数据下载步骤。
- 显式展示时间展开：避免把时间维完全藏进框架黑盒中。
- 输出要有教学对应关系：每张图都必须对应一个明确概念，而不是只做装饰。

## 推荐实现方案

采用“主实验 + 辅助分析”的拆分方式，对齐第 5 周的结构：

### 方案选择

候选方案有三类：

1. 单脚本方案：一个脚本包掉训练、采样、隐藏状态可视化和长序列分析
2. 主实验 + 辅助分析方案
3. 高度框架封装方案：主要依赖 `nn.RNN`

最终选择方案 2，原因如下：

- 比单脚本方案更清晰，主线实验和现象分析不会混在一起。
- 比高度封装方案更适合第 6 周教学，因为 `hidden state`、`time step`、`参数共享` 都可以显式展示。
- 风格上最接近第 5 周：一个主脚本负责核心概念，一个辅助脚本负责扩展理解。

## 目录结构设计

新增目录：

`experiments/rnn-and-sequence-modeling/`

目录内文件如下：

- `README.md`
  - 面向学习者解释本周主题、目录结构、运行方式、运行后该重点观察什么。
- `char_rnn_next_char.py`
  - 本周主实验脚本。
  - 使用仓库内置短英文语料完成字符级 next-character prediction。
- `rnn_sequence_length_probe.py`
  - 辅助分析脚本。
  - 对比不同序列长度下的训练难度、梯度范数和最终效果。
- `rnn_advantages_and_limits.md`
  - 一页总结文档。
  - 说明 RNN 的优势、适用场景和关键局限。
- `rnn_unrolled_computation_graph.svg`
  - 展开后的结构图 / 计算图。
  - 说明 `x_t`、`h_{t-1}`、`h_t`、`y_t`、参数共享与 BPTT。
- `run_all.py`
  - 本周统一入口。
- `outputs/`
  - 脚本运行后保存图像与文本结果。

## 主实验设计

### 文件：`char_rnn_next_char.py`

这个脚本负责讲清楚“字符级 RNN 是如何逐步处理序列的”。

### 语料选择

使用仓库内置的短小英文语料，不依赖外部下载。语料由脚本内部构造，内容围绕本周主题，例如：

- `recurrent models read one token at a time.`
- `hidden states carry context across time steps.`
- `long sequences are harder to optimize with simple rnn models.`

语料要求：

- 词表小，便于字符级展示
- 语义与本周概念强相关
- 能重复若干次，让训练曲线稳定但运行时间不过长

### 代码模块划分

主脚本建议按以下顺序组织：

1. `build_demo_corpus()`
   - 构造小型英文语料
2. `CharTokenizer`
   - 把字符映射成 `token ids`
   - 与第 5 周形成连续性
3. `build_sequence_dataset(context_length)`
   - 用滑动窗口构造训练样本
   - 每个时间步预测下一个字符
4. `CharRNNModel`
   - 结构为 `Embedding -> RNNCell -> Linear vocab head`
5. `forward_sequence()`
   - 显式按时间步循环
   - 保存每一步 hidden state
6. `train_model()`
   - 标准 PyTorch 训练循环
   - 记录 loss、accuracy、gradient norm
   - 使用 gradient clipping 保证教学脚本稳定运行
7. `sample_text()`
   - 训练后从模型采样字符序列
8. `predict_next_chars()`
   - 固定若干上下文，展示 top-k 下一个字符预测
9. 可视化函数
   - 输出训练曲线、hidden state 热力图、预测轨迹图

### 为什么使用 `nn.RNNCell`

不直接使用 `nn.RNN` 作为主要教学实现，原因如下：

- 第 6 周的重点是“显式看懂时间展开”
- `RNNCell` 更适合把每一步的状态更新写清楚
- 便于记录 `hidden state trajectory`
- 便于把“参数在每个时间步共享”说清楚

这并不否定 `nn.RNN` 的价值，只是教学上更适合把基础机制先摊开讲。

## 辅助分析实验设计

### 文件：`rnn_sequence_length_probe.py`

这个脚本只做一件事：对比不同序列长度时 RNN 的训练难度。

设计思路：

- 固定模型骨架不变
- 固定语料分布不变
- 只改变序列长度或 BPTT 展开长度
- 对比不同设置下的训练 loss、最终准确率和 gradient norm

这个拆分的价值在于：

- 主实验聚焦“RNN 怎么工作”
- 辅助实验聚焦“RNN 为什么会在长依赖上变难”

避免把两类信息混在一个脚本里，导致学习主线发散。

## 输出产物设计

本周默认产出 4 张图和 1 个文本结果：

### 1. `outputs/char_rnn_training_curves.png`

内容：

- 训练 loss 曲线
- 可叠加 gradient norm 曲线

目的：

- 说明字符级 RNN 确实可以学会基础 next-character prediction
- 把训练稳定性与梯度现象联系起来

### 2. `outputs/rnn_hidden_state_dynamics.png`

内容：

- 固定一段文本
- 按时间步可视化 hidden state 热力图

目的：

- 让 `hidden state` 从抽象概念变成可观察对象
- 说明模型状态会随着输入字符不断更新

### 3. `outputs/rnn_prediction_trace.png`

内容：

- 固定若干上下文
- 可视化 top-k 的下一个字符预测概率

目的：

- 展示“模型读到当前位置后，如何基于当前 hidden state 预测下一个字符”
- 帮助学习者把训练任务和推理行为联系起来

### 4. `outputs/rnn_sequence_length_comparison.png`

内容：

- 对比短序列、中等序列、较长序列下的 loss / accuracy / gradient norm

目的：

- 直观说明序列越长，训练越容易变难
- 对应 BPTT、梯度消失和梯度爆炸的直觉

### 5. `outputs/rnn_generated_samples.txt`

内容：

- 训练后用不同 temperature 采样的字符片段

目的：

- 让学习者看到模型已经学会局部字符模式
- 同时也能看到它在长距离一致性上的不足

## 结构图设计

### 文件：`rnn_unrolled_computation_graph.svg`

图中应明确表达以下元素：

- 输入字符 `x_t`
- embedding 输出 `e_t`
- 前一时刻隐藏状态 `h_{t-1}`
- 当前隐藏状态 `h_t`
- 输出 logits `o_t`
- 预测目标 `y_t`
- 共享参数：
  - `W_xh`
  - `W_hh`
  - `W_hy`
- 时间方向展开
- 反向传播沿时间反向展开，标记为 `BPTT`

设计目标不是做复杂美术，而是让学习者能一眼看出：

- 为什么 RNN 是“同一个单元反复使用”
- 为什么它能处理变长序列
- 为什么训练时要把梯度沿时间展开回传

## 文档设计

### 文件：`README.md`

README 要完成四件事：

1. 把第 5 周和第 6 周串起来
   - 第 5 周解决“文本怎么变成向量”
   - 第 6 周解决“模型怎么按顺序处理这些向量”
2. 解释目录结构和运行方式
3. 告诉学习者每个文件负责什么
4. 明确运行后应该重点观察什么

README 需要重点强调：

- 为什么本周选字符级 RNN
- 为什么使用 `RNNCell` 展开时间步
- 为什么加入辅助分析脚本
- 为什么序列长度会影响训练难度

### 文件：`rnn_advantages_and_limits.md`

这份文档作为一页总结，聚焦以下内容：

- 优势
  - 能处理变长输入
  - 参数共享
  - 按顺序建模自然
- 局限
  - 难并行
  - 长依赖困难
  - BPTT 链条长
  - 梯度消失 / 爆炸

收尾要自然过渡到第 7 周：

- 为什么后来需要 LSTM / GRU
- 第 7 周将如何在第 6 周基础上扩展

## 统一入口设计

### 文件：`run_all.py`

统一入口按前几周风格编写：

- 先运行 `char_rnn_next_char.py`
- 输出分隔线
- 再运行 `rnn_sequence_length_probe.py`

这样目录行为与第 5 周保持一致，降低学习者的使用成本。

## 测试与验证要求

本周实现完成后，需要满足以下验证标准：

1. `run_all.py` 能一键运行并成功输出图像和文本到 `outputs/`
2. 主脚本单独运行时也能独立完成训练和保存结果
3. README 与代码产物一致，不出现文件名不匹配或输出说明缺失
4. 每张图片都能清楚对应一个概念：
   - 训练有效性
   - hidden state 动态
   - next-character 预测行为
   - 长序列训练困难
5. 代码注释保持教学友好，但不过度啰嗦
6. 运行时间控制在教学环境可接受范围内，不依赖 GPU

## 风险与取舍

### 风险 1：字符级建模结果可能过于简单

取舍：

- 这是有意为之
- 第 6 周的目标是理解序列建模机制，而不是追求复杂文本质量

### 风险 2：显式时间展开会让代码比 `nn.RNN` 更长

取舍：

- 代码略长，但换来教学透明度
- 这符合本周“先看懂原理，再谈封装”的目标

### 风险 3：梯度消失 / 爆炸不一定在每次运行都特别戏剧化

取舍：

- 通过辅助分析脚本放大序列长度差异
- 通过记录 gradient norm 提供更直接的现象证据

## 实施范围边界

第 6 周不做以下内容：

- 不引入 LSTM / GRU 对照实验
- 不引入注意力机制
- 不做复杂语料或外部数据下载
- 不做多层 RNN、双向 RNN、beam search 等扩展

这些内容应留给第 7 周及之后，避免当前一周失焦。

## 预期结果

完成后，第 6 周会与前五周形成清晰连续的学习链：

- 第 5 周：文本如何数值化
- 第 6 周：序列如何按时间步建模
- 第 7 周：为什么要用更强的门控结构改进 RNN

学习者完成本周后，应至少能回答：

1. hidden state 在 RNN 里到底扮演什么角色？
2. 为什么同一个 RNN 单元可以处理不同长度的序列？
3. BPTT 和普通反向传播相比，多了哪一层“沿时间展开”的含义？
4. 为什么序列变长后，vanilla RNN 更容易变难训练？
5. 为什么第 7 周会自然引出 LSTM / GRU？
