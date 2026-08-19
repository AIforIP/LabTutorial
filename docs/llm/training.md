---
title: 训练一个语言模型
description: 从数据批次到参数更新的完整训练循环
---

# 3. 训练一个语言模型

模型定义完成后，我们需要持续提供训练批次、计算下一 token 预测损失，并用反向传播更新参数。

## 训练目标

给定 token 序列 $x_1, x_2, \ldots, x_T$，自回归语言模型学习最大化：

$$
p(x_1, \ldots, x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t})
$$

实践中通常最小化所有位置上的交叉熵损失。

## 最小训练循环

```python
model.train()

for input_ids, labels in dataloader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = criterion(logits.flatten(0, 1), labels.flatten())
    loss.backward()
    optimizer.step()
```

真实训练还需要学习率调度、梯度裁剪、检查点保存、混合精度与日志记录。但在加入这些组件前，应先确认最小循环能够在一个很小的数据集上过拟合。

## 先做过拟合测试

取一个固定的小批次重复训练。如果损失无法明显下降，问题往往来自标签错位、因果掩码、参数未注册或优化器配置。这个测试比直接启动长时间训练更有效。

## 观察什么

- 训练损失是否持续下降；
- 梯度范数是否突然变大或变为 `NaN`；
- 每秒处理的 token 数是否稳定；
- 验证集损失何时停止改善。

## 下一步

后续章节将继续补充数据工程、并行训练、模型评测与推理优化。
