---
title: BPE Tokenizer
description: 从直观示例到大规模语料，完整实现 BPE Tokenizer 的训练过程
---

# 1. BPE Tokenizer

语言模型并不直接读取文本。Tokenizer 会把字符串转换为整数序列；模型完成计算后，它还要把整数序列还原成人能读懂的文字。

本节从一个刻意保持简单的实现开始，先弄清 Byte Pair Encoding（BPE）的训练逻辑，再将同一套思路扩展到更大的真实语料。

> 本节整理自 [CS336 Assignment 1: BPE Tokenizer's Detailed Implementation](https://wangqiyao.me/blogs/2025/CS336/Assignment1/BPE/)，并针对本课程的章节结构调整了说明和代码。

## 1.1 BPE 在做什么

如果按字符切分，词表很小，但输入序列会很长；如果按完整单词切分，常见词很紧凑，却无法自然处理新词。BPE 选择折中的办法：从很小的基础词表出发，反复合并语料中最常出现的相邻 token 对。

训练过程可以概括成三步：

1. **初始化词表**：用 `0–255` 表示所有单字节，再加入 `<|endoftext|>` 等特殊 token；
2. **预分词**：先把文本划分为较小片段，避免跨越不合理的边界进行合并；
3. **反复合并**：统计相邻 token 对，选择频率最高的一对合并，并把结果加入词表。

重复第三步，直到词表达到指定大小。训练最终得到两个结果：`vocab` 负责 token ID 与字节串的映射，`merges` 记录所有合并规则及其先后顺序。

## 1.2 一个便于理解的版本

先使用一段很小的训练文本：

```python
text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""
```

这个实现不追求速度。它的价值是让每一步都可见，适合用来检查自己是否真正理解了算法。

### 初始化词表

现代 BPE Tokenizer 通常在字节层面工作。这样只需 256 个基础 token，就能表示任意 UTF-8 文本，也不会遇到真正意义上的未知字符。

```python
def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    return vocab
```

这里的值使用 `bytes`，而不是已经解码的字符。一个中文字通常由多个 UTF-8 字节组成，过早解码会让字节级算法变得混乱。

### 预分词

教学示例可以先按空白切分，并记录每个片段出现了多少次：

```python
from collections import Counter


def pre_tokenize(text: str) -> dict[str, int]:
    return dict(Counter(text.split()))
```

例如 `low` 出现五次。之后统计 `l-o`、`o-w` 时，应把这五次全部计入，而不是只处理一次。

真实 Tokenizer 不宜直接使用 `split()`。它无法精细区分标点、数字与空白，也会丢失部分边界信息。大规模版本中，我们会换成正则预分词。

### 统计相邻 token 对

把每个预分词片段表示为 token 元组，并结合它在语料中的出现次数完成统计：

```python
from collections import Counter


def count_pairs(
    words: dict[tuple[bytes, ...], int],
) -> Counter[tuple[bytes, bytes]]:
    counts = Counter()

    for word, frequency in words.items():
        for left, right in zip(word, word[1:]):
            counts[(left, right)] += frequency

    return counts
```

需要注意两个边界：长度小于 2 的片段没有相邻对；同一个片段可能包含多个相同的 token 对，它们都应被统计。

### 执行一次合并

找到目标 token 对后，要在每个预分词片段内从左向右扫描。匹配时消费两个 token，否则只消费一个。

```python
def merge_pair(
    word: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    merged = []
    i = 0

    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(word[i])
            i += 1

    return tuple(merged)
```

合并必须局限在单个预分词片段内部。例如 `dog!` 是否允许把 `g` 与 `!` 合并，应由预分词规则决定，而不能让 BPE 的全局扫描偶然决定。

### 组合训练循环

现在可以把前面的步骤组合起来：

```python
def train_toy_bpe(text: str, num_merges: int):
    vocab = init_vocab(["<|endoftext|>"])
    word_counts = pre_tokenize(text)

    words = {
        tuple(bytes([byte]) for byte in word.encode("utf-8")): count
        for word, count in word_counts.items()
    }
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        pair_counts = count_pairs(words)
        if not pair_counts:
            break

        # CS336 约定：频率相同时选择字典序较大的 token 对。
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        words = {
            merge_pair(word, best_pair): count
            for word, count in words.items()
        }

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

    return vocab, merges
```

`merges` 的顺序是模型的一部分。编码新文本时，如果多个规则都能匹配，必须按照训练得到的优先级应用，不能只看最终字符串长度。

::: tip 先验证，再优化
打印每轮选中的 `best_pair` 和更新后的 `words`。对于小语料，你应当能手工复核前两三轮的结果。
:::

## 1.3 面向真实语料的训练

简单版本每轮都会遍历所有片段，并重新计算所有 token 对。当语料扩大到 TinyStories 或 OpenWebText 时，这种方式会非常慢。要保持算法含义不变，同时减少重复工作。

### 按特殊 token 划分文件

大文件可以并行预处理，但不能在任意字节位置切开，否则可能破坏 UTF-8 字符或跨文档边界。CS336 提供的思路是先估计均匀位置，再向后寻找 `<|endoftext|>`，把它作为安全边界。

```python
import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    for boundary_index in range(1, len(boundaries) - 1):
        position = boundaries[boundary_index]
        file.seek(position)

        while True:
            block = file.read(4096)
            if block == b"":
                boundaries[boundary_index] = file_size
                break

            offset = block.find(split_special_token)
            if offset != -1:
                boundaries[boundary_index] = position + offset
                break

            position += len(block)

    return sorted(set(boundaries))
```

边界可能重合，所以最后需要去重；实际返回的 chunk 数量可能少于请求数量。

### 使用正则完成预分词

先移除特殊 token，再通过 GPT 系列 Tokenizer 常用的模式划分普通文本：

```python
import regex as re


PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"
    r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def pre_tokenize_chunk(text: str, special_tokens: list[str]):
    special_pattern = "|".join(re.escape(token) for token in special_tokens)
    documents = re.split(special_pattern, text)

    result: list[list[bytes]] = []
    for document in documents:
        for match in re.finditer(PATTERN, document):
            token = match.group(0).encode("utf-8")
            result.append([bytes([byte]) for byte in token])

    return result
```

`re.escape()` 很重要。特殊 token 中的 `|`、`[` 等字符可能具有正则含义，未经转义会改变匹配结果。

### 只更新受影响的位置

最大的优化来自增量更新。除了维护 `pair -> count`，再维护一个倒排索引 `pair -> pre-token indices`：

```python
from collections import defaultdict

pair_counts = defaultdict(int)
pair_to_indices = defaultdict(set)

for token_index, token in enumerate(pre_tokens):
    for left, right in zip(token, token[1:]):
        pair = (left, right)
        pair_counts[pair] += 1
        pair_to_indices[pair].add(token_index)
```

每轮选出 `best_pair` 后，只处理 `pair_to_indices[best_pair]` 中列出的片段：

1. 删除这些片段原来贡献的相邻对计数；
2. 在片段中执行合并；
3. 把合并后的相邻对重新加入计数和倒排索引。

这样仍然会重复扫描受影响片段，但不再为一次局部合并遍历整份语料。

::: warning 容易遗漏的细节
读取受影响索引时应先复制集合，例如 `affected = pair_to_indices[best_pair].copy()`。更新计数会同时修改原集合，直接迭代可能漏掉元素或触发运行时错误。
:::

## 1.4 性能为什么仍可能不够好

原实现处理 TinyStories 仍需要数小时，说明“使用多进程”不等于已经解决性能问题。建议分别计时以下阶段：

- 文件读取与 UTF-8 解码；
- 正则预分词；
- 初始 token 对统计；
- 每轮最高频 token 对的选择；
- 受影响片段的删除、合并与重新计数。

常见瓶颈是每轮线性寻找最高频 token 对，以及 Python 对象、集合和字典带来的大量开销。后续可以尝试优先队列、惰性失效、更加紧凑的数据表示，或将热点路径移到更低层实现。

## 1.5 检查你的实现

完成训练代码后，至少验证以下情况：

1. 空文件与只含特殊 token 的文件不会崩溃；
2. 频率相同时，合并结果符合规定的字典序规则；
3. 合并不会跨越预分词边界和特殊 token；
4. 中文、emoji 等多字节字符可以先编码、再无损解码；
5. 单进程与多进程产生相同的 `vocab` 和 `merges`；
6. 最终词表大小不超过 `vocab_size`。

## 本节小结

BPE 训练的核心只有“统计相邻对并合并”一句话，但一个可用实现还必须处理字节表示、预分词边界、特殊 token、稳定的 tie-break 规则与大规模语料性能。

下一节将使用训练得到的 token ID，构建 Decoder-only Transformer 的输入与模型主体。

## 参考资料

- [原始博客：BPE Tokenizer's Detailed Implementation](https://wangqiyao.me/blogs/2025/CS336/Assignment1/BPE/)
- [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics)
