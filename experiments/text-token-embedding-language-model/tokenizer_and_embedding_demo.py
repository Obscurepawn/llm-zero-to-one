#!/usr/bin/env python3
"""
第 5 周：文本、Token、Embedding 和语言模型任务。

这份脚本的目标不是做一个“真实可商用”的 tokenizer 或语言模型，
而是把第 5 周最重要的几条主线串起来，让初学者能顺着代码把概念看懂：

1. 原始文本为什么不能直接送进神经网络
2. 文本是怎么被切成 token、再映射成 token id 的
3. one-hot 和 embedding 在表示上到底有什么差别
4. 语言模型里“根据前文预测下一个 token”到底长什么样
5. 为什么顺序信息对序列建模非常重要

整个脚本大致分成两部分：

- 第 1 部分：字符级 tokenizer + next-token 数据集 + one-hot / embedding 对比
- 第 2 部分：一个顺序敏感的小实验，说明“丢掉顺序”会出什么问题

建议阅读顺序：

1. `CharTokenizer`
2. `build_next_token_dataset`
3. `save_tokenizer_preview`
4. `plot_one_hot_vs_embedding`
5. `build_order_sensitive_dataset`
6. `BagOfEmbeddingsLM` 与 `PositionAwareLM`
7. `train_order_model`
8. `main`
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# 使用无界面的 Agg 后端。
# 这样即使在纯终端、服务器或没有图形界面的环境里，也能正常保存图片。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# 某些环境里装的是带 CUDA 支持的 torch，但机器驱动并不真的可用。
# 这里我们只是做一个小教学实验，不想让这些环境噪声干扰输出日志，
# 所以把相关警告静掉。
warnings.filterwarnings("ignore", message="CUDA initialization:.*")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - runtime guard
    # 如果用户机器上没有安装 torch，就直接给出清晰报错，
    # 避免后面在更深的位置才抛出难懂的异常。
    raise SystemExit(
        "当前环境缺少 torch。请先执行 `python -m pip install torch matplotlib numpy`。"
    ) from exc


# 当前脚本所在目录。
ROOT_DIR = Path(__file__).resolve().parent

# 所有输出文件都统一放到 outputs/ 里，和前几周的结构保持一致。
OUTPUT_DIR = ROOT_DIR / "outputs"


def ensure_output_dir() -> None:
    """
    确保输出目录存在。

    这类教学脚本通常会保存图片、文本预览等结果，
    所以在真正开始运行前，先把目录准备好。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """
    固定随机种子。

    目的不是“保证每一次都百分之百完全一样”，
    而是尽量让：

    - embedding 初始化
    - 数据打乱顺序
    - 小模型训练结果

    在多次运行时保持接近，方便教学和调试。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def visible_token(token: str) -> str:
    """
    把一些“肉眼不容易看见”的字符，转成更适合展示的形式。

    例如：
    - 空格 `" "` 显示成 `<space>`
    - 换行 `"\n"` 显示成 `<newline>`

    这样在输出词表和样本预览时，你能明确看到这些字符的存在。
    """
    if token == " ":
        return "<space>"
    if token == "\n":
        return "<newline>"
    return token


class CharTokenizer:
    """
    一个最小字符级 tokenizer。

    这里的“字符级”意思是：
    - 每个字符都是一个 token
    - 空格、换行、标点也都是 token

    这不是工业界最常见的大模型 tokenizer 形式，
    但它非常适合教学，因为逻辑最直接：

    文本 -> 字符 -> id

    你可以把它理解成“最小版本的文本数值化器”。
    """

    def __init__(
        self,
        text: str,
        special_tokens: tuple[str, ...] = ("<PAD>", "<UNK>"),
    ) -> None:
        """
        根据一段训练文本构造字符词表。

        参数说明：
        - text: 用来建立词表的原始文本
        - special_tokens: 保留的特殊 token

        默认使用两个特殊 token：
        - `<PAD>`: 以后如果需要把不同长度序列补齐，会用到它
        - `<UNK>`: 如果遇到词表外字符，就回退到它

        这里先做 `set(text)`，得到语料里出现过的所有不同字符；
        再做 `sorted(...)`，是为了让词表顺序稳定、可复现。
        """
        unique_tokens = sorted(set(text))

        # id_to_token: id -> token
        # 比如 5 -> "a"
        self.id_to_token = list(special_tokens) + unique_tokens

        # token_to_id: token -> id
        # 比如 "a" -> 5
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

        # 保存常用特殊 token，后面 decode 时会直接用到。
        self.pad_token = special_tokens[0]
        self.unk_token = special_tokens[1]

    @property
    def vocab_size(self) -> int:
        """
        返回词表大小。

        对 tokenizer 来说，词表大小是一个非常重要的量，
        因为：
        - one-hot 的维度等于词表大小
        - embedding 表的行数等于词表大小
        - 模型输出层的类别数通常也和词表大小一致
        """
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        """
        把字符串编码成 token id 列表。

        例如：
            "cat" -> [7, 5, 12]

        如果某个字符不在词表里，就回退成 `<UNK>` 对应的 id。
        这也是 special token 存在的一个核心用途。
        """
        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(char, unk_id) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """
        把 token id 列表再还原回字符串。

        参数：
        - token_ids: 一串整数 id
        - skip_special_tokens:
          - False: 特殊 token 也照常拼回去
          - True: 遇到 `<PAD>` / `<UNK>` 这类 token 就跳过

        这一步对应 tokenizer 的“反向过程”：
            id -> token -> 文本
        """
        tokens: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token[token_id]

            # 有些场景下我们不希望在重建文本时看到特殊 token，
            # 比如只想看可读内容本身，这时就可以选择跳过。
            if skip_special_tokens and token in {self.pad_token, self.unk_token}:
                continue

            tokens.append(token)

        # 对字符级 tokenizer 来说，直接把字符拼起来就能得到原文。
        return "".join(tokens)

    def token_frequencies(self, text: str) -> dict[str, int]:
        """
        统计文本里每个 token 的出现次数。

        注意：因为这是字符级 tokenizer，
        所以这里统计的其实就是“每个字符出现了多少次”。
        """
        frequencies: dict[str, int] = {}
        for char in text:
            frequencies[char] = frequencies.get(char, 0) + 1
        return frequencies


def build_demo_corpus() -> str:
    """
    构造一份很小的字符级语料。

    这里没有去加载外部大语料，而是手工写几句和本周主题强相关的英文句子。
    这样做有几个好处：

    1. 语料内容你一眼能看懂
    2. 词表规模较小，便于观察
    3. 句子中反复出现了 token / embedding / language model / order 等关键词，
       便于后面分析 next-token 样本

    最后把这几句重复 6 次，是为了让统计更稳定一些，同时又不至于太大。
    """
    lines = [
        "text becomes tokens before it enters a neural net.",
        "embedding layers map token ids to dense vectors.",
        "language models predict the next token from context.",
        "word order changes meaning in sequence data.",
        "a tokenizer turns text into integers a model can read.",
    ]
    return "\n".join(lines * 6)


def build_next_token_dataset(
    token_ids: list[int],
    context_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用滑动窗口生成 next-token 训练样本。

    这是语言模型里最经典的任务形式：

        给一段前文 context
        预测它后面的下一个 token

    比如当 `context_size=4` 时：
        [t0, t1, t2, t3] -> t4
        [t1, t2, t3, t4] -> t5

    这里输出两个数组：
    - contexts: 每一行是一段固定长度的上下文
    - targets: 每个上下文对应要预测的“下一个 token”
    """
    contexts: list[list[int]] = []
    targets: list[int] = []

    # end_idx 表示“目标 token”的位置。
    # 那么它前面的 context_size 个 token，就是当前样本的输入上下文。
    for end_idx in range(context_size, len(token_ids)):
        contexts.append(token_ids[end_idx - context_size : end_idx])
        targets.append(token_ids[end_idx])

    # 最后转成 NumPy 数组，便于后面统一处理。
    return np.asarray(contexts, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def save_tokenizer_preview(
    tokenizer: CharTokenizer,
    corpus: str,
    context_size: int,
    contexts: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
) -> None:
    """
    把 tokenizer 和 next-token 数据集的关键信息保存成一份文本预览。

    这份预览文件是给“人”看的，不是给模型看的。
    它主要帮助你回答这些问题：

    - 当前词表里到底有哪些 token
    - 每个 token 的 id 是多少
    - 哪些字符最常出现
    - 一段文本被 encode 之后长什么样
    - next-token 训练样本具体是怎么构造出来的
    """
    # 先统计语料里的 token 频率，并取出现次数最多的前 20 个。
    frequencies = tokenizer.token_frequencies(corpus)
    top_tokens = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:20]

    # preview_lines 用来逐行收集最终要写到文件里的内容。
    preview_lines = [
        "Week 5 tokenizer and language-model preview",
        "=" * 48,
        "",
        f"Vocabulary size: {tokenizer.vocab_size}",
        f"Corpus length (characters): {len(corpus)}",
        f"Context size: {context_size}",
        "",
        "First 30 vocabulary entries:",
    ]

    # 把词表前 30 项写进去，方便你看到 id 和 token 的对应关系。
    for token_id, token in list(enumerate(tokenizer.id_to_token))[:30]:
        preview_lines.append(f"  id={token_id:>2d} token={visible_token(token)}")

    preview_lines.extend(
        [
            "",
            "Top token frequencies:",
        ]
    )

    # 把最常见的 token 及其出现次数写进去。
    for token, count in top_tokens:
        preview_lines.append(f"  token={visible_token(token):>10s} count={count}")

    # 这里选一段简单文本做编码 / 解码演示。
    sample_text = "language models"
    preview_lines.extend(
        [
            "",
            f"Sample text: {sample_text!r}",
            f"Encoded ids: {tokenizer.encode(sample_text)}",
            f"Decoded text: {tokenizer.decode(tokenizer.encode(sample_text))!r}",
            "",
            "First 12 next-token samples:",
        ]
    )

    # 把前 12 个 next-token 样本打印出来。
    # 这一步很关键，因为它能让你直接看到：
    #
    #   context_ids / context_text -> target_id / target_text
    #
    # 也就是语言模型任务到底在学什么。
    for index in range(12):
        context_ids = contexts[index].tolist()
        target_id = int(targets[index])
        preview_lines.append(
            "  "
            f"context_ids={context_ids} "
            f"context_text={tokenizer.decode(context_ids)!r} "
            f"-> target_id={target_id} "
            f"target_text={tokenizer.decode([target_id])!r}"
        )

    # 把整份预览写到文本文件里。
    output_path.write_text("\n".join(preview_lines), encoding="utf-8")


def plot_one_hot_vs_embedding(
    tokenizer: CharTokenizer,
    sample_text: str = "text.",
    embedding_dim: int = 8,
) -> tuple[Path, float]:
    """
    画出 one-hot 和 embedding lookup 的差别。

    这段函数最重要的目的，不是“画一张图”，而是帮助你真正想明白：

    1. 什么是 one-hot
    2. 为什么 one-hot 虽然能用，但不太适合直接拿来做大模型输入表示
    3. embedding 到底是什么
    4. 为什么 embedding lookup 和 one-hot 乘矩阵，本质上是同一件事

    先说 one-hot：

    - 假设词表大小 vocab_size = 30000
    - 那么每个 token 都会对应一个长度为 30000 的向量
    - 向量里只有 1 个位置是 1，其余 29999 个位置都是 0

    这就叫“高维、稀疏”表示。

    例如 token_id = 2，词表大小 = 5 时：

        one_hot(2) = [0, 0, 1, 0, 0]

    为什么说它“不太适合直接用”？

    - 维度会随着词表变大而变得很大
    - 里面绝大多数值都是 0，存储和计算都比较浪费
    - 它只能表达“我是第几个 token”，不能自然表达“这个 token 和别的 token 是否相似”

    再说 embedding：

    embedding 可以理解成一张“可训练的向量表”。

    假设：
    - vocab_size = 5
    - embedding_dim = 3

    那么 embedding_weight 这张表的形状就是：

        [5, 3]

    你可以把它想成：

        embedding_weight =
        [
          [0.1, 0.2, 0.3],   # token 0 的向量
          [0.4, 0.5, 0.6],   # token 1 的向量
          [0.7, 0.8, 0.9],   # token 2 的向量
          [1.0, 1.1, 1.2],   # token 3 的向量
          [1.3, 1.4, 1.5],   # token 4 的向量
        ]

    这时：

        embedding_lookup(2)

    的意思就是：

        直接把 embedding_weight 的第 2 行取出来

    结果是：

        [0.7, 0.8, 0.9]

    而如果你先把 token_id=2 变成 one-hot：

        [0, 0, 1, 0, 0]

    再乘 embedding_weight：

        [0, 0, 1, 0, 0] @ embedding_weight

    结果仍然是：

        [0.7, 0.8, 0.9]

    原因很简单：
    one-hot 里只有第 2 位是 1，所以矩阵乘法时，
    本质上就是把 embedding_weight 的第 2 行“选出来”。

    所以这里要验证的数学事实是：

        embedding_lookup(token_id)
        ==
        one_hot(token_id) @ embedding_weight

    左边是“查表”写法，
    右边是“矩阵乘法”写法，
    但两者的数值结果相同。

    这也解释了为什么现代框架一般直接使用 embedding lookup：
    它和 one-hot 乘矩阵在数学上等价，但在工程上更高效、更自然。
    """
    # 先把一段示例文本编码成 token id。
    token_ids = tokenizer.encode(sample_text)

    # np.eye(vocab_size) 会生成一个 vocab_size x vocab_size 的单位矩阵。
    #
    # 例如 vocab_size=5 时：
    #
    # np.eye(5) =
    # [
    #   [1, 0, 0, 0, 0],
    #   [0, 1, 0, 0, 0],
    #   [0, 0, 1, 0, 0],
    #   [0, 0, 0, 1, 0],
    #   [0, 0, 0, 0, 1],
    # ]
    #
    # 这里的每一行，其实就是“某个 token 对应的 one-hot 向量”。
    # 比如：
    # - 第 0 行 -> token_id=0 的 one-hot
    # - 第 2 行 -> token_id=2 的 one-hot = [0, 0, 1, 0, 0]
    #
    # 接下来这一句：
    #     np.eye(tokenizer.vocab_size)[token_ids]
    # 的含义不是“取一个元素”，而是“用 token_ids 这组下标，去批量取多行”。
    #
    # 也就是说，如果：
    #     token_ids = [2, 4, 1]
    #
    # 那么：
    #     np.eye(5)[token_ids]
    #
    # 就等价于：
    #     [
    #       np.eye(5)[2],
    #       np.eye(5)[4],
    #       np.eye(5)[1],
    #     ]
    #
    # 最终得到的结果会是一个二维矩阵：
    # [
    #   [0, 0, 1, 0, 0],   # token_id = 2
    #   [0, 0, 0, 0, 1],   # token_id = 4
    #   [0, 1, 0, 0, 0],   # token_id = 1
    # ]
    #
    # 你可以把它理解成：
    # - 单位矩阵的“每一行”都是一个 one-hot 模板
    # - token_ids 这个列表告诉 NumPy：“按这个顺序，把对应的那些行拿出来”
    #
    # 所以这里的 [] 不是普通 Python list 的“单个位置取值”语义，
    # 而是 NumPy 的“高级索引（advanced indexing）”：
    # 当 [] 里面放的是一个整数列表 / 数组时，表示一次性取出多行。
    one_hot = np.eye(tokenizer.vocab_size, dtype=np.float32)[token_ids]

    # 这里创建一个 PyTorch 的 embedding 层。
    #
    # 它内部其实就是一张形状为：
    #   [vocab_size, embedding_dim]
    # 的参数表。
    torch.manual_seed(7)
    embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)

    # 为了演示，手动把 embedding 权重初始化到 [-1, 1] 区间。
    # 这样图里的颜色变化会更明显。
    with torch.no_grad():
        embedding.weight.uniform_(-1.0, 1.0)

    # embedding lookup：直接用 token id 去查表。
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    embedding_vectors = embedding(token_tensor).detach().cpu().numpy()

    # 把 embedding 权重矩阵取出来，后面要验证：
    # one_hot @ W 和 embedding lookup 是否相同。
    embedding_weight = embedding.weight.detach().cpu().numpy()

    # 这一步就是“one-hot 乘嵌入矩阵”。
    one_hot_projection = one_hot @ embedding_weight

    # 计算两种方式得到的最大绝对误差。
    # 理论上这里应该非常接近 0。
    max_difference = float(np.max(np.abs(embedding_vectors - one_hot_projection)))

    # 一共画 3 张子图：
    # - 左：one-hot
    # - 中：embedding lookup
    # - 右：表示维度对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 第 1 张图：one-hot 矩阵
    # 横轴：token 在文本中的位置
    # 纵轴：词表维度
    axes[0].imshow(one_hot.T, aspect="auto", cmap="Blues", interpolation="nearest")
    axes[0].set_title(f"One-hot\nvocab_dim={tokenizer.vocab_size}")
    axes[0].set_xlabel("Token position")
    axes[0].set_ylabel("Vocabulary index")
    axes[0].set_xticks(range(len(token_ids)))
    axes[0].set_xticklabels([visible_token(char) for char in sample_text])

    # 第 2 张图：embedding 向量
    # 横轴还是 token 位置，纵轴变成 embedding 的维度。
    axes[1].imshow(embedding_vectors.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axes[1].set_title(f"Embedding lookup\nembed_dim={embedding_dim}")
    axes[1].set_xlabel("Token position")
    axes[1].set_ylabel("Embedding dimension")
    axes[1].set_xticks(range(len(token_ids)))
    axes[1].set_xticklabels([visible_token(char) for char in sample_text])

    # 用柱状图对比表示维度：
    # - one-hot: 维度 = vocab_size
    # - embedding: 维度 = embedding_dim
    representation_dims = [tokenizer.vocab_size, embedding_dim]

    # density 粗略表示“非零元素占比”。
    # 对 one-hot 来说通常非常稀疏，对 embedding 来说通常更稠密。
    densities = [
        1.0 / tokenizer.vocab_size,
        float(np.count_nonzero(embedding_vectors) / embedding_vectors.size),
    ]
    labels = ["one-hot", "embedding"]
    bars = axes[2].bar(labels, representation_dims, color=["#4C78A8", "#F58518"])
    axes[2].set_title("Representation size")
    axes[2].set_ylabel("Vector dimension")

    # 在柱子上方标出 density，帮助对比“稀疏 vs 稠密”。
    for bar, density in zip(bars, densities, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.6,
            f"density={density:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(f"Sample text: {sample_text!r}", fontsize=12)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "one_hot_vs_embedding_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path, max_difference


def build_order_sensitive_dataset(
    repeats: int = 120,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], list[tuple[list[str], str]]]:
    """
    构造一个顺序敏感的 next-token 数据集。

    这个小实验的设计重点是：

    - ['a', 'b'] -> 'x'
    - ['b', 'a'] -> 'y'

    你会发现：
    - 两个上下文里出现的 token 集合是一样的，都是 {a, b}
    - 唯一不同的是顺序

    所以如果一个模型只看“有哪些 token”，不看“顺序”，
    那它就没法把这两个样本区分开。

    这正是我们后面想证明的事情。
    """
    patterns = [
        (["a", "b"], "x"),
        (["b", "a"], "y"),
        (["c", "d"], "m"),
        (["d", "c"], "n"),
    ]

    # 先从所有 pattern 中收集出 vocabulary。
    #
    # 这里故意不用“一行写完”的推导式，而是拆成更直观的几步：
    # 1. 先把所有出现过的 token 收集到一个集合里
    # 2. 集合会自动去重
    # 3. 最后再排序，保证 token -> id 的顺序稳定
    vocabulary_set: set[str] = set()
    for context, target in patterns:
        for token in context:
            vocabulary_set.add(token)
        vocabulary_set.add(target)

    vocabulary = sorted(vocabulary_set)
    token_to_id = {token: idx for idx, token in enumerate(vocabulary)}

    contexts: list[list[int]] = []
    targets: list[int] = []
    examples: list[tuple[list[str], str]] = []

    # 把 pattern 重复多次，生成一份足够训练的小数据集。
    for _ in range(repeats):
        for context_tokens, target_token in patterns:
            contexts.append([token_to_id[token] for token in context_tokens])
            targets.append(token_to_id[target_token])
            examples.append((context_tokens, target_token))

    # 打乱样本顺序，避免模型按固定顺序“背答案”。
    order = np.random.permutation(len(contexts))
    context_array = np.asarray(contexts, dtype=np.int64)[order]
    target_array = np.asarray(targets, dtype=np.int64)[order]
    shuffled_examples = [examples[index] for index in order]

    # 这里直接返回 torch.Tensor，
    # 因为后面训练小模型时会直接交给 DataLoader。
    return (
        torch.tensor(context_array, dtype=torch.long),
        torch.tensor(target_array, dtype=torch.long),
        token_to_id,
        shuffled_examples,
    )


def split_train_val(
    contexts: torch.Tensor,
    targets: torch.Tensor,
    train_ratio: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    把顺序实验数据切分成训练集和验证集。

    这里故意保持最简单的切分方式：
    - 前 80% 当训练集
    - 后 20% 当验证集

    对这个教学 demo 来说已经足够。
    """
    split_index = int(len(contexts) * train_ratio)
    return (
        contexts[:split_index],
        targets[:split_index],
        contexts[split_index:],
        targets[split_index:],
    )


# ---------------------------------------------------------------------------
# Shape 推导表
# ---------------------------------------------------------------------------
#
# 下面两个模型最容易混淆的地方，就是“每一层输入输出到底是什么 shape”。
# 所以在正式看 BagOfEmbeddingsLM / PositionAwareLM 之前，先把整条链路列成表。
#
# 一、先约定几个符号
# ---------------------------------------------------------------------------
# batch_size:
#   一个 mini-batch 里有多少个样本
#
# context_size:
#   每个样本里有多少个 token
#
# embedding_dim:
#   每个 token 被 embedding 层映射成多少维向量
#
# vocab_size:
#   词表大小，也就是“下一个 token 的候选类别数”
#
# 在当前这个顺序实验里，具体数值通常是：
# - context_size = 2
# - embedding_dim = 8
# - vocab_size = 8
# - batch_size 在训练时通常是 32，在验证时通常是 64
#
# 二、输入到模型前的数据 shape
# ---------------------------------------------------------------------------
# context_ids.shape = [batch_size, context_size]
#
# 语义解释：
# - 第 0 维：batch 里的第几个样本
# - 第 1 维：该样本里第几个 token
#
# 例如：
# [
#   [0, 1],   # ['a', 'b']
#   [1, 0],   # ['b', 'a']
#   [2, 3],   # ['c', 'd']
# ]
#
# 如果当前 batch_size = 32，那么就是：
# [32, 2]
#
# 三、经过 embedding 之后的 shape
# ---------------------------------------------------------------------------
# embeddings = self.embedding(context_ids)
# embeddings.shape = [batch_size, context_size, embedding_dim]
#
# 语义解释：
# - 第 0 维：batch 里的第几个样本
# - 第 1 维：该样本中第几个 token 位置
# - 第 2 维：这个 token 向量的第几维
#
# 在当前实验里，如果 batch_size = 32：
# [32, 2, 8]
#
# 四、BagOfEmbeddingsLM 的 shape 变化
# ---------------------------------------------------------------------------
# Step 1:
# context_ids                           -> [batch_size, 2]
#
# Step 2:
# embeddings = self.embedding(...)      -> [batch_size, 2, 8]
#
# Step 3:
# pooled = embeddings.mean(dim=1)       -> [batch_size, 8]
#
# 为什么？
# - dim=1 对应“序列位置”这一维
# - 对 dim=1 做平均，表示把两个 token 的向量逐维平均
# - 平均之后，位置维被消掉，只剩一个上下文向量
#
# Step 4:
# logits = self.classifier(pooled)      -> [batch_size, 8]
#
# 为什么？
# - self.classifier = nn.Linear(embedding_dim, vocab_size)
# - 也就是 nn.Linear(8, 8)
# - 线性层只改最后一维，所以最后一维从 8 变成 vocab_size=8
#
# 最终：
# logits.shape = [batch_size, vocab_size]
#
# 五、PositionAwareLM 的 shape 变化
# ---------------------------------------------------------------------------
# Step 1:
# context_ids                           -> [batch_size, 2]
#
# Step 2:
# embeddings = self.embedding(...)      -> [batch_size, 2, 8]
#
# Step 3:
# flattened = embeddings.reshape(...)   -> [batch_size, 16]
#
# 为什么？
# - 原来单个样本是 [2, 8]
# - reshape 后把它展平成 [16]
# - 16 = context_size * embedding_dim = 2 * 8
#
# Step 4:
# logits = self.classifier(flattened)   -> [batch_size, 8]
#
# 为什么？
# - self.classifier = nn.Linear(context_size * embedding_dim, vocab_size)
# - 也就是 nn.Linear(16, 8)
# - 输出仍然是对词表 8 个候选 token 的打分
#
# 最终：
# logits.shape = [batch_size, vocab_size]
#
# 六、两个模型最大的区别
# ---------------------------------------------------------------------------
# BagOfEmbeddingsLM:
# - [batch_size, 2, 8] --mean(dim=1)--> [batch_size, 8]
# - “位置维”被平均掉了，所以顺序信息容易丢失
#
# PositionAwareLM:
# - [batch_size, 2, 8] --reshape--> [batch_size, 16]
# - 位置 0 的向量和位置 1 的向量仍然被分别保留，只是拼接成了一行
# - 因此顺序信息还在
#
# 七、和 loss 的接口如何对齐
# ---------------------------------------------------------------------------
# target_ids.shape = [batch_size]
# logits.shape     = [batch_size, vocab_size]
#
# 这正好符合 CrossEntropyLoss 的接口要求：
# - logits: 每个样本对每个类别的分数
# - target_ids: 每个样本真实类别的 id
#
class BagOfEmbeddingsLM(nn.Module):
    """
    一个故意忽略顺序的模型。

    结构非常简单：

        context_ids
        -> embedding lookup
        -> 对所有位置求平均
        -> 线性分类

    它的问题就在“求平均”这一步。

    因为一旦把多个位置的 embedding 平均掉：
    - ['a', 'b']
    - ['b', 'a']

    得到的平均向量完全一样。

    所以它天然无法区分顺序相反但 token 集合相同的序列。
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()

        # 词表大小决定 embedding 表有多少行。
        # embedding_dim 决定每个 token 被映射成多少维的向量。
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 平均池化之后，得到一个 embedding_dim 维向量；
        # 再通过一个线性层，输出对每个词表 token 的打分 logits。
        self.classifier = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        # context_ids 形状大致是 [batch_size, context_size]
        embeddings = self.embedding(context_ids)

        # 沿着序列维做平均，把整个上下文压成一个向量。
        pooled = embeddings.mean(dim=1)

        # 返回 logits，后面交给 CrossEntropyLoss。
        return self.classifier(pooled)


class PositionAwareLM(nn.Module):
    """
    一个保留顺序信息的模型。

    它和前一个模型的关键差别在于：
    - 不把所有位置平均掉
    - 而是保留每个位置自己的 embedding

    这里的做法很直接：

        context_ids
        -> embedding lookup
        -> 把每个位置的 embedding 直接拼接起来
        -> 线性分类

    对长度固定、上下文很短的教学实验来说，
    这已经足够表现出“顺序”这一因素的重要性。
    """

    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 因为我们保留了每个位置的 embedding，
        # 所以输入给线性层的总维度 = context_size * embedding_dim。
        self.classifier = nn.Linear(context_size * embedding_dim, vocab_size)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(context_ids)

        # reshape 后的 flattened 会把多个位置的 embedding 按顺序拼成一个长向量。
        # 这样第 1 个位置和第 2 个位置就不再被“混在一起”。
        flattened = embeddings.reshape(context_ids.size(0), -1)
        return self.classifier(flattened)


@dataclass
class TrainingHistory:
    """
    保存训练过程中的曲线数据。

    这样后面画图时，就不需要把几个列表分散传来传去。
    """

    name: str
    train_loss: list[float]
    val_accuracy: list[float]


def evaluate_accuracy(model: nn.Module, data_loader: DataLoader) -> float:
    """
    在验证集上评估准确率。

    这里不更新梯度，只做前向推理。
    """
    model.eval()
    total_correct = 0
    total_examples = 0

    # with torch.no_grad() 表示：
    # 这里只是评估，不需要构建反向传播计算图，能省内存也更快。
    with torch.no_grad():
        for context_ids, target_ids in data_loader:
            logits = model(context_ids)
            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == target_ids).sum().item())
            total_examples += target_ids.size(0)

    return total_correct / total_examples


def train_order_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str,
    epochs: int = 60,
    learning_rate: float = 0.08,
) -> TrainingHistory:
    """
    训练一个最小 next-token 模型。

    这段代码非常适合建立“标准训练循环”的直觉：

    1. 取一批数据
    2. 前向计算 logits
    3. 计算 loss
    4. backward
    5. optimizer.step()
    6. 每轮结束后做一次验证
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses: list[float] = []
    val_accuracies: list[float] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for context_ids, target_ids in train_loader:
            # 清空上一轮 batch 留下的梯度。
            optimizer.zero_grad()

            # 前向传播：得到每个样本对各个 token 的打分 logits。
            logits = model(context_ids)

            # CrossEntropyLoss 会自动做：
            # logits -> softmax -> 与真实标签比较
            loss = criterion(logits, target_ids)

            # 反向传播，计算梯度。
            loss.backward()

            # 用优化器更新参数。
            optimizer.step()

            # 统计整轮训练的总损失。
            total_loss += float(loss.item()) * target_ids.size(0)
            total_examples += target_ids.size(0)

        average_loss = total_loss / total_examples
        val_accuracy = evaluate_accuracy(model, val_loader)
        train_losses.append(average_loss)
        val_accuracies.append(val_accuracy)

        # 不必每一轮都打印，挑几轮关键节点输出即可。
        if epoch in {0, 1, 2, 4, 9, 19, 39, epochs - 1}:
            print(
                f"[{name}] "
                f"epoch={epoch + 1:3d}/{epochs} "
                f"train_loss={average_loss:.4f} "
                f"val_acc={val_accuracy:.4f}"
            )

    return TrainingHistory(name=name, train_loss=train_losses, val_accuracy=val_accuracies)


def plot_position_importance(histories: list[TrainingHistory]) -> Path:
    """
    画出“忽略顺序 vs 保留顺序”的训练对比图。

    左图：
    - 训练 loss 随 epoch 的变化

    右图：
    - 验证准确率随 epoch 的变化

    这张图的目标是让你一眼看出：
    顺序信息对于某些任务不是“可有可无”，而是决定模型能不能学会任务。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for history in histories:
        axes[0].plot(history.train_loss, label=history.name)
        axes[1].plot(history.val_accuracy, label=history.name)

    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()

    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.suptitle("Why token order matters for next-token prediction", fontsize=12)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "position_order_importance.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def append_order_experiment_summary(
    output_path: Path,
    examples: list[tuple[list[str], str]],
    histories: list[TrainingHistory],
) -> None:
    """
    把顺序实验的摘要追加到 tokenizer 预览文件中。

    这样最终只看一个文本文件，也能同时看到：
    - tokenizer / 数据集预览
    - 顺序实验样例
    - 两个模型的最终验证准确率
    """
    summary_lines = [
        "",
        "Order-sensitive next-token task",
        "=" * 48,
        "",
        "Example patterns:",
    ]

    # 先把几条示例 pattern 写进去。
    for context_tokens, target_token in examples[:8]:
        summary_lines.append(f"  context={context_tokens} -> target={target_token}")

    summary_lines.extend(
        [
            "",
            "Final validation accuracy:",
        ]
    )

    # 再把各模型最终准确率写进去。
    for history in histories:
        summary_lines.append(f"  {history.name}: {history.val_accuracy[-1]:.4f}")

    # 用追加模式打开文件，不覆盖之前已经写好的 tokenizer 预览。
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))
        handle.write("\n")


def main() -> None:
    """
    运行完整的第 5 周教学实验。

    你可以把整个 main 看成一条完整的数据流：

    1. 准备输出目录与随机种子
    2. 构造原始文本语料
    3. 用 CharTokenizer 建词表并编码
    4. 构造 next-token 训练样本
    5. 保存 tokenizer / 数据集预览
    6. 画 one-hot vs embedding 对比图
    7. 构造顺序敏感数据集
    8. 训练两个小模型做对照
    9. 保存顺序实验结果
    """
    ensure_output_dir()
    set_seed(42)

    # 第一步：准备一份演示语料。
    corpus = build_demo_corpus()

    # 第二步：用语料建立字符级 tokenizer。
    tokenizer = CharTokenizer(corpus)

    # 第三步：把整份文本编码成 token id 序列。
    token_ids = tokenizer.encode(corpus)

    # 第四步：构造语言模型训练样本。
    # 这里把每个 context 的长度固定成 6。
    context_size = 6
    contexts, targets = build_next_token_dataset(token_ids, context_size=context_size)

    # 第五步：把 tokenizer、词表、样本预览保存成文本文件。
    preview_path = OUTPUT_DIR / "tokenizer_dataset_preview.txt"
    save_tokenizer_preview(
        tokenizer=tokenizer,
        corpus=corpus,
        context_size=context_size,
        contexts=contexts,
        targets=targets,
        output_path=preview_path,
    )

    # 第六步：画 one-hot 与 embedding 的对比图。
    # 同时得到两种表示方式数值等价的最大误差。
    one_hot_path, max_difference = plot_one_hot_vs_embedding(tokenizer=tokenizer)
    print(f"Saved one-hot vs embedding figure to: {one_hot_path}")
    print(f"Max difference between one-hot @ W and embedding lookup: {max_difference:.8f}")

    # 第七步：构造一个顺序敏感的小型 next-token 数据集。
    order_contexts, order_targets, token_to_id, examples = build_order_sensitive_dataset()

    # 第八步：切训练集 / 验证集。
    train_x, train_y, val_x, val_y = split_train_val(order_contexts, order_targets)

    # 第九步：封装成 DataLoader。
    # - 训练集打乱
    # - 验证集不打乱
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64, shuffle=False)

    # 第十步：训练两个模型做对照。
    #
    # bag_of_embeddings:
    #   只看 token 集合，不保留顺序
    #
    # position_aware:
    #   保留每个位置的 embedding，因此能分辨顺序
    histories = [
        train_order_model(
            model=BagOfEmbeddingsLM(vocab_size=len(token_to_id), embedding_dim=8),
            train_loader=train_loader,
            val_loader=val_loader,
            name="bag_of_embeddings",
        ),
        train_order_model(
            model=PositionAwareLM(vocab_size=len(token_to_id), embedding_dim=8, context_size=2),
            train_loader=train_loader,
            val_loader=val_loader,
            name="position_aware",
        ),
    ]

    # 第十一步：保存训练对比图。
    position_path = plot_position_importance(histories)

    # 第十二步：把顺序实验摘要追加到预览文件。
    append_order_experiment_summary(preview_path, examples, histories)

    # 最后打印输出文件位置和最终结果。
    print(f"Saved tokenizer preview to: {preview_path}")
    print(f"Saved position importance figure to: {position_path}")
    print("Final validation accuracy:")
    for history in histories:
        print(f"- {history.name}: {history.val_accuracy[-1]:.4f}")


if __name__ == "__main__":
    # 只有当这个文件被“直接运行”时，才执行 main()。
    # 如果它只是被别的文件 import，则不会自动跑实验。
    main()
