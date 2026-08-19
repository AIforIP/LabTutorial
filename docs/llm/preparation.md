---
title: 准备工作
description: LLM from Scratch 的环境与前置知识
---

# 0. 准备工作

这一节只完成一件事：准备一个可复现、容易排查问题的实验环境。暂时不追求复杂的工程配置。

## 前置知识

你需要能够阅读基础 Python，并理解矩阵乘法、导数和概率分布。对 PyTorch 不熟悉没有关系，后续会结合模型逐步介绍。

## 推荐环境

- Python 3.11 或更高版本
- PyTorch 2.x
- 一块支持 CUDA 的 GPU（入门章节也可先使用 CPU）
- Git 与常用命令行工具

## 最小检查

安装 PyTorch 后运行：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())

x = torch.randn(2, 3)
w = torch.randn(3, 4)
print((x @ w).shape)
```

最后一行应输出 `torch.Size([2, 4])`。这同时也是后续最常见的调试习惯：每经过一个模块，都确认张量的形状是否符合预期。

## 接下来

环境准备好后，我们先处理语言模型真正看到的输入——token，而不是字符或单词。
