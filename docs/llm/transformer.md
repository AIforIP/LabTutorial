---
title: Transformer
description: 理解并实现 Decoder-only Transformer
---

# 2. Transformer

这一章从张量形状出发，搭建一个 Decoder-only Transformer。我们先理解数据如何流动，再讨论具体优化。

## 整体数据流

输入 token 的形状通常是 `(batch, sequence)`。经过嵌入层后变为 `(batch, sequence, hidden)`，随后依次通过多个 Transformer block，最终投影为每个位置上的词表概率。

```text
token ids
   ↓ embedding
hidden states
   ↓ attention + MLP × N
contextualized states
   ↓ output projection
next-token logits
```

## 自注意力

注意力机制让每个位置根据当前上下文聚合信息。对输入 $X$ 做三个线性投影得到 Query、Key 和 Value：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

缩放点积注意力为：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

对语言模型而言，还需要因果掩码，确保当前位置不能看到未来 token。

## 残差连接与归一化

残差连接提供更短的梯度路径，归一化则控制中间激活的尺度。实现时应明确采用 Pre-Norm 还是 Post-Norm；两者的计算顺序不同，不能混用权重。

## 动手

先实现单头注意力并打印每一步的张量形状。确认无误后，再将 hidden 维度拆成多个 head。若一开始就写完整模块，维度错误通常更难定位。

## 本章小结

Transformer 的每个组件都不复杂，难点在于它们之间精确的数据契约。下一章会把模型、数据与优化器放进同一个训练循环。
