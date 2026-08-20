---
title: 深度学习基础
description: 梯度下降、反向传播、神经网络与 CNN、RNN、Transformer
---

# 4. 深度学习基础

深度学习使用多层可微函数从数据中学习表示。理解它需要同时关注三个层面：张量怎样变化、参数怎样通过梯度更新、不同架构对数据结构作了什么假设。

## 4.1 张量与计算图

张量可以看作多维数组。每个操作都应明确输入与输出形状：

```python
import torch

x = torch.randn(32, 128)   # batch=32, features=128
w = torch.randn(128, 64)
b = torch.randn(64)
y = x @ w + b

assert y.shape == (32, 64)
```

大量深度学习错误本质上是形状、设备或数据类型错误。养成检查 `shape`、`dtype` 和 `device` 的习惯。

## 4.2 梯度下降

设损失为 $\mathcal{L}(\theta)$，最基本的梯度下降更新为：

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal{L}(\theta_t),
$$

其中 $\eta$ 是学习率。学习率过大可能震荡或发散，过小则训练缓慢。实际常用 SGD with Momentum、Adam 或 AdamW，但任何优化器都不能补救错误的数据和损失实现。

## 4.3 反向传播

反向传播不是一种新的求导规则，而是把链式法则高效应用到计算图。若：

$$
z=f(x),\qquad y=g(z),
$$

则：

$$
\frac{\partial y}{\partial x}
=\frac{\partial y}{\partial z}\frac{\partial z}{\partial x}.
$$

框架在前向传播时记录运算关系，反向传播时从损失开始，按相反顺序累积梯度。

```python
x = torch.tensor(2.0, requires_grad=True)
y = (x ** 2 + 1) ** 3
y.backward()
print(x.grad)
```

建议手算导数，再与自动微分结果比较。这样能理解 `requires_grad`、叶子张量和梯度累积，而不是把 `.backward()` 当作魔法。

## 4.4 损失函数

### 均方误差

回归任务常用：

$$
\mathcal{L}_{\text{MSE}}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2.
$$

### 交叉熵

多分类任务常用：

$$
\mathcal{L}_{\text{CE}}=-\sum_{k=1}^{K}y_k\log p_k.
$$

PyTorch 的 `CrossEntropyLoss` 接收未归一化 logits，并在内部完成 `log_softmax`。提前手工执行 softmax 不仅多余，还可能降低数值稳定性。

## 4.5 激活函数

如果多层网络只有线性变换，多层仍可合并成一个线性变换。激活函数提供非线性表达能力。

- **Sigmoid**：输出在 $(0,1)$，适合概率门控，但深层网络可能梯度饱和；
- **Tanh**：输出零中心，仍可能饱和；
- **ReLU**：计算简单，是经典默认选择；
- **GELU / SiLU**：更平滑，Transformer 中常见。

选择激活函数时关注梯度、输出范围、计算成本和架构惯例。

## 4.6 全连接网络

全连接层为：

$$
h=\phi(Wx+b).
$$

多层感知机通过重复线性层和非线性层学习复杂映射。它不显式利用图像的局部结构或序列顺序，因此也是理解 CNN、RNN 和 Transformer 为什么出现的起点。

## 4.7 CNN

卷积神经网络使用局部连接和权重共享：同一个卷积核在不同位置检测相似模式。它特别适合网格结构数据，如图像，也曾广泛用于文本局部特征抽取。

需要理解：

- 卷积核、步幅、填充与感受野；
- 通道数与特征图；
- 池化和下采样；
- 参数共享为何减少参数量。

## 4.8 RNN、LSTM 与 GRU

RNN 按时间步递归更新隐藏状态：

$$
h_t=\phi(W_xx_t+W_hh_{t-1}+b).
$$

它天然表达顺序，但长序列训练容易出现梯度消失或爆炸，并且时间步难以并行。

LSTM 通过输入门、遗忘门和输出门控制信息流；GRU 用更少的门控结构实现类似目标。学习时重点理解“门如何控制保留和更新”，不必先背诵所有公式。

## 4.9 Transformer

Transformer 通过注意力直接建立序列位置间的联系，避免 RNN 的逐步依赖。核心缩放点积注意力为：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
$$

在进入大语言模型前，需要理解：

- Query、Key、Value 的含义和形状；
- 多头注意力如何拆分表示空间；
- 位置编码为什么必要；
- 残差连接、归一化和前馈网络；
- padding mask 与 causal mask 的区别。

这里的目标是理解架构，不要求从零优化高性能实现。

## 4.10 一个完整训练循环

```python
model.train()

for inputs, targets in dataloader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
```

训练时同时监控训练损失、验证指标、学习率、梯度范数、显存和吞吐量。先尝试在一个很小的数据批次上过拟合，这是检查模型和标签是否正确的高效方法。

## 4.11 常见问题

- 损失不下降：检查标签、学习率、参数是否加入优化器；
- 出现 `NaN`：检查数据、除零、对数输入和梯度爆炸；
- 训练很好而验证差：检查过拟合和数据泄漏；
- GPU 利用率低：检查数据加载、批量大小和频繁的 CPU–GPU 同步；
- 结果无法复现：固定种子并记录环境，但也要理解某些 GPU 算子仍可能非确定。

## 本章任务

使用 PyTorch 在 FashionMNIST 或类似数据集上完成：

1. 训练一个两层 MLP；
2. 改为简单 CNN，并比较参数量与准确率；
3. 绘制训练与验证损失；
4. 展示至少十个错误样本并分析原因；
5. 手工计算一个两层标量网络的梯度，与 autograd 对照。

## 推荐学习

- [李宏毅 Machine Learning 2017 Fall](https://speech.ee.ntu.edu.tw/~hylee/ml/2017-fall.php)：深度学习、反向传播、CNN、RNN 等课程材料
- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)：从数据加载到训练与保存模型的官方教程
