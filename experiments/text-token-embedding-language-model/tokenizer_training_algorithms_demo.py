#!/usr/bin/env python3
"""
第 5 周补充：几种常见 tokenizer “训练”方式的最小 demo。

这份脚本把下面四类 tokenizer 放到同一个文件里：

1. 字符级 tokenizer
   - 几乎不需要训练
   - 只需要从语料里收集出现过的字符

2. BPE
   - 反复合并最频繁的相邻符号对
   - 更像频率驱动的压缩过程

3. WordPiece
   - 教学版 demo 使用常见的关联分数近似
   - 用它演示“为什么不是只看频率”

4. Unigram
   - 用一个真正的概率模型做 EM 风格训练
   - 直接最小化语料负对数似然

这不是工业级 tokenizer 实现，而是“能让你把训练过程看清楚”的教学代码。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


def ensure_output_dir() -> None:
    """确保输出目录存在。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_demo_corpus() -> list[str]:
    """
    构造一份很小的训练语料。

    这里故意选带有词形变化的英文单词，方便观察：
    - low / lower / lowest
    - new / newer / newest
    - hug / hugged / hugging
    - run / runner / running
    """
    return [
        "low low low lower lowest",
        "new new new newer newest",
        "wide wide wider widest",
        "hug hug hug hugs hugged hugging",
        "run run run runs runner running",
        "token token token tokens tokenizer tokenizing",
    ]


def build_word_frequencies(corpus_lines: list[str]) -> Counter[str]:
    """把小语料转成词频统计。"""
    frequencies: Counter[str] = Counter()
    for line in corpus_lines:
        for word in line.strip().split():
            frequencies[word] += 1
    return frequencies


def format_counter(counter: Counter[str]) -> list[str]:
    """把词频表格式化成可读文本。"""
    lines: list[str] = []
    for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {token:<12s} count={count}")
    return lines


def logsumexp(values: list[float]) -> float:
    """稳定计算 log(sum(exp(values)))."""
    if not values:
        return float("-inf")
    max_value = max(values)
    if math.isinf(max_value):
        return max_value
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


@dataclass
class CharTokenizerResult:
    vocabulary: list[str]
    encoded_sample: list[int]
    decoded_sample: str


def train_char_tokenizer(corpus_lines: list[str], sample_text: str) -> CharTokenizerResult:
    """
    训练字符级 tokenizer。

    实际上它没有传统意义上的 loss 或优化过程：
    - 只是把出现过的字符收集起来
    - 给每个字符分配一个 id
    """
    joined_text = "\n".join(corpus_lines)
    vocabulary = ["<PAD>", "<UNK>"] + sorted(set(joined_text))
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    unk_id = token_to_id["<UNK>"]
    encoded = [token_to_id.get(char, unk_id) for char in sample_text]
    decoded = "".join(vocabulary[token_id] for token_id in encoded if vocabulary[token_id] != "<PAD>")
    return CharTokenizerResult(vocabulary=vocabulary, encoded_sample=encoded, decoded_sample=decoded)


@dataclass
class BPEResult:
    merges: list[tuple[str, str]]
    merge_stats: list[tuple[tuple[str, str], int]]
    vocabulary: list[str]


def build_bpe_state(word_freqs: Counter[str]) -> dict[str, list[str]]:
    """把每个词初始化成字符序列。"""
    return {word: list(word) for word in word_freqs}


def count_pairs(state: dict[str, list[str]], word_freqs: Counter[str]) -> Counter[tuple[str, str]]:
    """统计所有相邻符号对的频率。"""
    pair_counts: Counter[tuple[str, str]] = Counter()
    for word, symbols in state.items():
        frequency = word_freqs[word]
        for left, right in zip(symbols[:-1], symbols[1:], strict=True):
            pair_counts[(left, right)] += frequency
    return pair_counts


def merge_pair_in_state(
    state: dict[str, list[str]],
    target_pair: tuple[str, str],
    merged_token: str,
) -> dict[str, list[str]]:
    """把指定相邻对在所有词里合并成一个新符号。"""
    left_target, right_target = target_pair
    new_state: dict[str, list[str]] = {}

    for word, symbols in state.items():
        merged_symbols: list[str] = []
        index = 0
        while index < len(symbols):
            if (
                index < len(symbols) - 1
                and symbols[index] == left_target
                and symbols[index + 1] == right_target
            ):
                merged_symbols.append(merged_token)
                index += 2
            else:
                merged_symbols.append(symbols[index])
                index += 1
        new_state[word] = merged_symbols

    return new_state


def encode_with_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """按学习到的 merge 顺序，把一个词编码成 BPE 子词。"""
    symbols = list(word)
    for left, right in merges:
        merged_token = left + right
        updated_symbols: list[str] = []
        index = 0
        while index < len(symbols):
            if index < len(symbols) - 1 and symbols[index] == left and symbols[index + 1] == right:
                updated_symbols.append(merged_token)
                index += 2
            else:
                updated_symbols.append(symbols[index])
                index += 1
        symbols = updated_symbols
    return symbols


def train_bpe(word_freqs: Counter[str], num_merges: int = 16) -> BPEResult:
    """
    训练一个最小 BPE。

    核心逻辑非常简单：
    - 看哪一对相邻符号最常一起出现
    - 把它们合并
    - 重复很多次
    """
    state = build_bpe_state(word_freqs)
    merges: list[tuple[str, str]] = []
    merge_stats: list[tuple[tuple[str, str], int]] = []

    for _ in range(num_merges):
        pair_counts = count_pairs(state, word_freqs)
        if not pair_counts:
            break
        best_pair, best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        merged_token = best_pair[0] + best_pair[1]
        state = merge_pair_in_state(state, best_pair, merged_token)
        merges.append(best_pair)
        merge_stats.append((best_pair, best_count))

    vocabulary = sorted({symbol for symbols in state.values() for symbol in symbols})
    return BPEResult(merges=merges, merge_stats=merge_stats, vocabulary=vocabulary)


@dataclass
class WordPieceResult:
    merges: list[tuple[str, str]]
    merge_stats: list[tuple[tuple[str, str], float, int]]
    vocabulary: list[str]


def split_word_for_wordpiece(word: str) -> list[str]:
    """WordPiece 初始化：首字符裸露，后续字符加 ## 前缀。"""
    if not word:
        return []
    return [word[0]] + [f"##{char}" for char in word[1:]]


def merge_wordpiece_tokens(left: str, right: str) -> str:
    """
    合并两个 WordPiece token。

    例子：
    - h + ##u -> hu
    - hu + ##g -> hug
    - ##n + ##g -> ##ng
    """
    left_text = left[2:] if left.startswith("##") else left
    right_text = right[2:] if right.startswith("##") else right
    prefix = "##" if left.startswith("##") else ""
    return prefix + left_text + right_text


def build_wordpiece_state(word_freqs: Counter[str]) -> dict[str, list[str]]:
    """初始化 WordPiece 状态。"""
    return {word: split_word_for_wordpiece(word) for word in word_freqs}


def count_wordpiece_stats(
    state: dict[str, list[str]],
    word_freqs: Counter[str],
) -> tuple[Counter[str], Counter[tuple[str, str]]]:
    """统计当前 token 频率和相邻对频率。"""
    token_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()

    for word, pieces in state.items():
        frequency = word_freqs[word]
        for piece in pieces:
            token_counts[piece] += frequency
        for left, right in zip(pieces[:-1], pieces[1:], strict=True):
            pair_counts[(left, right)] += frequency

    return token_counts, pair_counts


def merge_wordpiece_state(
    state: dict[str, list[str]],
    target_pair: tuple[str, str],
) -> dict[str, list[str]]:
    """把最佳 pair 合并到整个语料状态里。"""
    left_target, right_target = target_pair
    merged_token = merge_wordpiece_tokens(left_target, right_target)
    new_state: dict[str, list[str]] = {}

    for word, pieces in state.items():
        merged_pieces: list[str] = []
        index = 0
        while index < len(pieces):
            if index < len(pieces) - 1 and pieces[index] == left_target and pieces[index + 1] == right_target:
                merged_pieces.append(merged_token)
                index += 2
            else:
                merged_pieces.append(pieces[index])
                index += 1
        new_state[word] = merged_pieces

    return new_state


def encode_with_wordpiece(word: str, vocabulary: set[str]) -> list[str]:
    """
    用最长匹配把一个词切成 WordPiece。

    这是 WordPiece 推理时最常见的直觉：
    每次尽量匹配当前能找到的最长子词。
    """
    pieces: list[str] = []
    start = 0

    while start < len(word):
        best_piece: str | None = None
        for end in range(len(word), start, -1):
            substring = word[start:end]
            candidate = substring if start == 0 else f"##{substring}"
            if candidate in vocabulary:
                best_piece = candidate
                break

        if best_piece is None:
            if start == 0:
                pieces.append("<UNK>")
                break
            pieces.append(f"##{word[start]}")
            start += 1
            continue

        pieces.append(best_piece)
        start += len(best_piece[2:] if best_piece.startswith("##") else best_piece)

    return pieces


def train_wordpiece(word_freqs: Counter[str], num_merges: int = 16) -> WordPieceResult:
    """
    训练一个教学版 WordPiece。

    这里用常见的近似分数：
        score(pair) = pair_count / (count(left) * count(right))

    直觉上，它更偏好“彼此强绑定”的 pair，而不是只看绝对频率。
    这不是工业实现的完整复刻，但足够解释核心想法。
    """
    state = build_wordpiece_state(word_freqs)
    merges: list[tuple[str, str]] = []
    merge_stats: list[tuple[tuple[str, str], float, int]] = []

    for _ in range(num_merges):
        token_counts, pair_counts = count_wordpiece_stats(state, word_freqs)
        if not pair_counts:
            break

        best_pair: tuple[str, str] | None = None
        best_score = float("-inf")
        best_count = 0

        for pair, pair_count in pair_counts.items():
            left_count = token_counts[pair[0]]
            right_count = token_counts[pair[1]]
            score = pair_count / (left_count * right_count)
            if score > best_score or (math.isclose(score, best_score) and pair_count > best_count):
                best_pair = pair
                best_score = score
                best_count = pair_count

        if best_pair is None:
            break

        state = merge_wordpiece_state(state, best_pair)
        merges.append(best_pair)
        merge_stats.append((best_pair, best_score, best_count))

    vocabulary = sorted({piece for pieces in state.values() for piece in pieces})
    return WordPieceResult(merges=merges, merge_stats=merge_stats, vocabulary=vocabulary)


@dataclass
class UnigramResult:
    probabilities: dict[str, float]
    training_history: list[tuple[int, int, float]]
    vocabulary: list[str]


def build_unigram_candidates(
    word_freqs: Counter[str],
    max_piece_length: int = 5,
    min_count: int = 2,
) -> tuple[dict[str, int], set[str]]:
    """
    从语料里收集候选子词。

    这里把所有出现足够多的子串都先当作候选，然后再交给 Unigram EM 去筛选。
    """
    candidate_counts: dict[str, int] = defaultdict(int)
    required_characters: set[str] = set()

    for word, frequency in word_freqs.items():
        required_characters.update(word)
        for start in range(len(word)):
            for end in range(start + 1, min(len(word), start + max_piece_length) + 1):
                piece = word[start:end]
                candidate_counts[piece] += frequency

    filtered_candidates = {
        piece: count
        for piece, count in candidate_counts.items()
        if count >= min_count or len(piece) == 1
    }
    return filtered_candidates, required_characters


def compute_unigram_word_logprob(word: str, log_probs: dict[str, float]) -> float:
    """用前向算法计算一个词在当前 Unigram 模型下的对数概率。"""
    length = len(word)
    # alpha[i] 表示：
    # “从词开头走到位置 i，所有合法切分路径的总 log probability”
    #
    # 例如 alpha[0] = 0，表示空前缀概率为 1（log(1)=0）。
    # 后面每次只要找到一个合法 piece，就可以把前缀路径继续延长。
    alpha = [float("-inf")] * (length + 1)
    alpha[0] = 0.0
    max_piece_length = max(len(piece) for piece in log_probs)

    for start in range(length):
        if math.isinf(alpha[start]):
            continue
        for end in range(start + 1, min(length, start + max_piece_length) + 1):
            piece = word[start:end]
            if piece in log_probs:
                # alpha[start] + log_probs[piece] 的含义：
                # 先用某种方式切到 start，再接上当前 piece，形成一条到达 end 的新路径。
                #
                # 因为在概率空间里：
                #   P(前缀路径 + 当前 piece) = P(前缀路径) * P(piece)
                # 所以到了 log 空间里，乘法就变成加法：
                #   log P(前缀路径 + 当前 piece)
                # = log P(前缀路径) + log P(piece)
                #
                # alpha[end] 原先可能已经有别的路径到达过，所以这里不能直接覆盖，
                # 而要用 logsumexp 把“旧路径概率”和“新路径概率”加起来。
                #
                # 也就是说：
                #   alpha[end] = log(所有到达 end 的路径概率之和)
                alpha[end] = logsumexp([alpha[end], alpha[start] + log_probs[piece]])

    return alpha[length]


# 下面给一个最小手算例子，帮助理解 alpha / beta / posterior 的语义。
#
# 假设词为 "abc"，候选 piece 及其概率为：
#   P("a")  = 0.30
#   P("b")  = 0.20
#   P("c")  = 0.10
#   P("ab") = 0.25
#   P("bc") = 0.15
#
# 那么所有合法切分有 3 种：
#   1. a + b + c   -> 0.30 * 0.20 * 0.10 = 0.006
#   2. ab + c      -> 0.25 * 0.10        = 0.025
#   3. a + bc      -> 0.30 * 0.15        = 0.045
#
# 所以整个词的总概率是：
#   P("abc") = 0.006 + 0.025 + 0.045 = 0.076
#
# 1. alpha 的含义
# alpha[i] 表示“从开头走到位置 i 的所有切分总概率”
#
#   位置:    0   1   2   3
#            | a | b | c |
#
#   alpha[0] = 1.0
#   alpha[1] = P("a") = 0.30
#   alpha[2] = P("ab") + P("a")P("b")
#            = 0.25 + 0.30*0.20
#            = 0.31
#   alpha[3] = alpha[2]P("c") + alpha[1]P("bc")
#            = 0.31*0.10 + 0.30*0.15
#            = 0.031 + 0.045
#            = 0.076
#
# 所以 alpha[3] 恰好就是整个词的总概率。
#
# 2. beta 的含义
# beta[i] 表示“从位置 i 走到结尾的所有切分总概率”
#
#   beta[3] = 1.0
#   beta[2] = P("c") = 0.10
#   beta[1] = P("b")P("c") + P("bc")
#           = 0.20*0.10 + 0.15
#           = 0.17
#   beta[0] = P("a")beta[1] + P("ab")beta[2]
#           = 0.30*0.17 + 0.25*0.10
#           = 0.051 + 0.025
#           = 0.076
#
# 所以 beta[0] 也等于整个词的总概率。
#
# 3. posterior 的含义
# 例如 piece = "ab"，它对应 span (0, 2)。
# 经过它的完整路径只有：
#   ab + c
#
# 因此：
#   posterior("ab" at 0:2)
#   = alpha[0] * P("ab") * beta[2] / P("abc")
#   = 1.0 * 0.25 * 0.10 / 0.076
#   = 0.025 / 0.076
#   ≈ 0.3289
#
# 这表示：在所有可能切分贡献的总概率质量里，大约 32.89% 的质量经过了 "ab"。
#
# 在代码里使用 log 概率时，会写成：
#   posterior_logprob = alpha[start] + log_probs[piece] + beta[end] - word_logprob
#
# 其中：
#   alpha[start]   -> log(前缀总概率)
#   log_probs[piece] -> log(piece 概率)
#   beta[end]      -> log(后缀总概率)
#   word_logprob   -> log(整词总概率)
#
# 之所以是加法，是因为在 log 空间里：
#   log(x * y * z) = log(x) + log(y) + log(z)
#
def expected_counts_for_word(
    word: str,
    word_frequency: int,
    log_probs: dict[str, float],
) -> tuple[dict[str, float], float]:
    """
    对单个词做前向后向，计算每个 piece 的期望计数。

    这一步是 Unigram 最关键的地方：
    - 不只看“一个最佳切分”
    - 而是对所有可能切分做求和
    """
    length = len(word)
    max_piece_length = max(len(piece) for piece in log_probs)
    # alpha[i]：
    # 从左往右，到达位置 i 的所有切分路径总 log probability。
    #
    # beta[i]：
    # 从位置 i 往右走到词尾的所有切分路径总 log probability。
    #
    # 两者合在一起，就能计算：
    # “某个 piece 出现在完整切分路径中的总概率”。
    alpha = [float("-inf")] * (length + 1)
    beta = [float("-inf")] * (length + 1)
    alpha[0] = 0.0
    beta[length] = 0.0

    for start in range(length):
        if math.isinf(alpha[start]):
            continue
        for end in range(start + 1, min(length, start + max_piece_length) + 1):
            piece = word[start:end]
            if piece in log_probs:
                # 前向转移：
                # 所有能走到 start 的前缀路径，都可以接上 piece 走到 end。
                alpha[end] = logsumexp([alpha[end], alpha[start] + log_probs[piece]])

    for start in range(length - 1, -1, -1):
        transitions: list[float] = []
        for end in range(start + 1, min(length, start + max_piece_length) + 1):
            piece = word[start:end]
            if piece in log_probs:
                # 后向转移：
                # 如果当前位置可以切出 piece，并且从 end 到词尾也有合法路径，
                # 那么这就是一条从 start 到词尾的合法后缀路径。
                transitions.append(log_probs[piece] + beta[end])
        beta[start] = logsumexp(transitions)

    # 整个词的总 log probability。
    # 它等于“所有合法切分路径概率求和”之后再取 log。
    word_logprob = alpha[length]
    piece_expectations: dict[str, float] = defaultdict(float)

    for start in range(length):
        for end in range(start + 1, min(length, start + max_piece_length) + 1):
            piece = word[start:end]
            if piece not in log_probs:
                continue
            # alpha[start] + log_probs[piece] + beta[end] 的含义：
            # 所有“经过当前 piece”的完整路径总分数。
            #
            # 其中：
            # - alpha[start]：从开头走到 piece 起点的总 log probability
            # - log_probs[piece]：当前 piece 自己的 log probability
            # - beta[end]：从 piece 终点走到词尾的总 log probability
            #
            # 再减去整个词的 word_logprob，相当于做归一化，
            # 得到“当前 piece 出现在一条随机切分路径中的后验概率”。
            posterior_logprob = alpha[start] + log_probs[piece] + beta[end] - word_logprob
            # 把这个后验概率乘上该词在语料里的出现次数，
            # 就是这个 piece 对全局期望计数的贡献。
            piece_expectations[piece] += word_frequency * math.exp(posterior_logprob)

    return dict(piece_expectations), word_frequency * (-word_logprob)


def viterbi_segment_unigram(word: str, log_probs: dict[str, float]) -> list[str]:
    """用最大概率路径给一个词做 Unigram 分词。"""
    length = len(word)
    best_score = [float("-inf")] * (length + 1)
    best_last_piece: list[str | None] = [None] * (length + 1)
    best_previous_index = [-1] * (length + 1)
    best_score[0] = 0.0
    max_piece_length = max(len(piece) for piece in log_probs)

    for start in range(length):
        if math.isinf(best_score[start]):
            continue
        for end in range(start + 1, min(length, start + max_piece_length) + 1):
            piece = word[start:end]
            if piece not in log_probs:
                continue
            candidate_score = best_score[start] + log_probs[piece]
            if candidate_score > best_score[end]:
                best_score[end] = candidate_score
                best_last_piece[end] = piece
                best_previous_index[end] = start

    pieces: list[str] = []
    index = length
    while index > 0 and best_last_piece[index] is not None:
        pieces.append(best_last_piece[index] or "")
        index = best_previous_index[index]
    pieces.reverse()
    return pieces


def train_unigram(
    word_freqs: Counter[str],
    target_vocab_size: int = 24,
    em_iterations_before_prune: int = 4,
    em_iterations_after_prune: int = 4,
) -> UnigramResult:
    """
    训练一个最小 Unigram tokenizer。

    训练逻辑：
    1. 先收集一个较大的候选子词集
    2. 用 EM 估计每个子词的概率
    3. 删掉贡献太小的子词
    4. 再继续 EM 微调
    """
    candidate_counts, required_characters = build_unigram_candidates(word_freqs)
    probabilities = {
        piece: count / sum(candidate_counts.values())
        for piece, count in candidate_counts.items()
    }
    training_history: list[tuple[int, int, float]] = []

    total_iterations = em_iterations_before_prune + em_iterations_after_prune

    for iteration in range(total_iterations):
        log_probs = {piece: math.log(probability) for piece, probability in probabilities.items()}
        expected_counts: dict[str, float] = defaultdict(float)
        total_negative_log_likelihood = 0.0

        for word, frequency in word_freqs.items():
            piece_expectations, word_nll = expected_counts_for_word(word, frequency, log_probs)
            total_negative_log_likelihood += word_nll
            for piece, expectation in piece_expectations.items():
                expected_counts[piece] += expectation

        if iteration == em_iterations_before_prune and len(probabilities) > target_vocab_size:
            removable_pieces = [
                piece
                for piece in probabilities
                if len(piece) > 1 and piece not in required_characters
            ]
            keep_budget = max(target_vocab_size - len(required_characters), 0)
            pieces_to_keep = sorted(
                removable_pieces,
                key=lambda piece: expected_counts.get(piece, 0.0),
                reverse=True,
            )[:keep_budget]
            new_vocabulary = set(required_characters) | set(pieces_to_keep)
            expected_counts = {
                piece: count
                for piece, count in expected_counts.items()
                if piece in new_vocabulary
            }

        total_expected = sum(expected_counts.values())
        probabilities = {
            piece: count / total_expected
            for piece, count in expected_counts.items()
            if count > 0.0
        }

        for character in required_characters:
            probabilities.setdefault(character, 1e-8)

        normalization = sum(probabilities.values())
        probabilities = {
            piece: probability / normalization
            for piece, probability in probabilities.items()
        }

        training_history.append((iteration + 1, len(probabilities), total_negative_log_likelihood))

    vocabulary = sorted(probabilities)
    return UnigramResult(
        probabilities=probabilities,
        training_history=training_history,
        vocabulary=vocabulary,
    )


def build_summary_lines(
    corpus_lines: list[str],
    word_freqs: Counter[str],
    char_result: CharTokenizerResult,
    bpe_result: BPEResult,
    wordpiece_result: WordPieceResult,
    unigram_result: UnigramResult,
) -> list[str]:
    """把四种 tokenizer demo 的结果整理成一份可读摘要。"""
    sample_words = ["lower", "lowest", "hugging", "running", "tokenizer", "tokenizing"]
    lines: list[str] = [
        "Tokenizer Training Algorithms Demo",
        "=" * 60,
        "",
        "Training corpus:",
    ]
    lines.extend([f"  {line}" for line in corpus_lines])
    lines.extend(["", "Word frequencies:"])
    lines.extend(format_counter(word_freqs))

    lines.extend(
        [
            "",
            "1. Character-level tokenizer",
            "-" * 60,
            "No optimization loss. Training just means collecting seen characters.",
            f"Vocabulary size: {len(char_result.vocabulary)}",
            f"First 24 vocab entries: {char_result.vocabulary[:24]}",
            "Sample word: 'tokenizer'",
            f"Encoded ids: {char_result.encoded_sample}",
            f"Decoded text: {char_result.decoded_sample!r}",
        ]
    )

    lines.extend(
        [
            "",
            "2. BPE",
            "-" * 60,
            "Training rule: repeatedly merge the most frequent adjacent pair.",
            "First 12 merges:",
        ]
    )
    for step, (pair, count) in enumerate(bpe_result.merge_stats[:12], start=1):
        merged_token = pair[0] + pair[1]
        lines.append(
            f"  step={step:>2d} pair={pair} count={count:<3d} -> merged={merged_token}"
        )
    lines.append("Sample segmentations:")
    for word in sample_words:
        lines.append(f"  {word:<12s} -> {encode_with_bpe(word, bpe_result.merges)}")

    lines.extend(
        [
            "",
            "3. WordPiece (teaching approximation)",
            "-" * 60,
            "Training rule here uses score(pair) = pair_count / (count(left) * count(right)).",
            "This is a simplified tutorial version, not an exact production trainer.",
            "First 12 merges:",
        ]
    )
    for step, (pair, score, count) in enumerate(wordpiece_result.merge_stats[:12], start=1):
        merged_token = merge_wordpiece_tokens(pair[0], pair[1])
        lines.append(
            f"  step={step:>2d} pair={pair} count={count:<3d} score={score:.6f} -> merged={merged_token}"
        )
    lines.append("Sample segmentations:")
    wordpiece_vocab = set(wordpiece_result.vocabulary)
    for word in sample_words:
        lines.append(f"  {word:<12s} -> {encode_with_wordpiece(word, wordpiece_vocab)}")

    lines.extend(
        [
            "",
            "4. Unigram",
            "-" * 60,
            "Objective: minimize corpus negative log-likelihood with EM-style updates.",
            "When pruning shrinks the vocabulary, NLL can jump briefly before the next EM rounds adapt.",
            "Training history:",
        ]
    )
    for iteration, vocab_size, nll in unigram_result.training_history:
        lines.append(f"  iter={iteration:>2d} vocab_size={vocab_size:<3d} total_nll={nll:.4f}")
    lines.append("Top pieces by probability:")
    top_pieces = sorted(
        unigram_result.probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:18]
    for piece, probability in top_pieces:
        lines.append(f"  piece={piece:<12s} prob={probability:.6f}")
    lines.append("Sample segmentations:")
    unigram_log_probs = {piece: math.log(prob) for piece, prob in unigram_result.probabilities.items()}
    for word in sample_words:
        lines.append(f"  {word:<12s} -> {viterbi_segment_unigram(word, unigram_log_probs)}")

    lines.extend(
        [
            "",
            "Quick comparison",
            "-" * 60,
            "Character-level: no loss, no merge search, one char = one token.",
            "BPE: greedy frequency merges, easy to implement, very intuitive.",
            "WordPiece: prefers strongly associated merges, often yields cleaner subwords.",
            "Unigram: probabilistic model with an explicit likelihood objective.",
        ]
    )
    return lines


def main() -> None:
    """运行四种 tokenizer 训练方式的统一 demo。"""
    ensure_output_dir()

    corpus_lines = build_demo_corpus()
    word_freqs = build_word_frequencies(corpus_lines)
    char_result = train_char_tokenizer(corpus_lines, sample_text="tokenizer")
    bpe_result = train_bpe(word_freqs)
    wordpiece_result = train_wordpiece(word_freqs)
    unigram_result = train_unigram(word_freqs)

    summary_lines = build_summary_lines(
        corpus_lines=corpus_lines,
        word_freqs=word_freqs,
        char_result=char_result,
        bpe_result=bpe_result,
        wordpiece_result=wordpiece_result,
        unigram_result=unigram_result,
    )

    output_path = OUTPUT_DIR / "tokenizer_training_algorithms_demo.txt"
    output_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved tokenizer training summary to: {output_path}")
    print("Highlights:")
    print(f"- Character vocab size: {len(char_result.vocabulary)}")
    print(f"- BPE merges learned: {len(bpe_result.merges)}")
    print(f"- WordPiece merges learned: {len(wordpiece_result.merges)}")
    print(f"- Unigram final vocab size: {len(unigram_result.vocabulary)}")
    print("- Unigram training history:")
    for iteration, vocab_size, nll in unigram_result.training_history:
        print(f"  iter={iteration:>2d} vocab_size={vocab_size:<3d} total_nll={nll:.4f}")


if __name__ == "__main__":
    main()
