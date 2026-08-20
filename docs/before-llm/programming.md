---
title: Python 与开发环境
description: Python、IDE、Git、Linux、SSH 与服务器使用基础
---

# 2. Python 与开发环境

学习编程最有效的方式是用它解决一个真实的小问题。语法只需要覆盖当前任务；当项目逐渐复杂，再学习函数、类、模块、测试与工程组织。

## 2.1 Python 必备内容

### 数据与控制流

首先熟悉：

- 数字、字符串、布尔值和 `None`；
- `list`、`tuple`、`dict`、`set`；
- `if`、`for`、`while`；
- 切片、列表推导式和字典推导式；
- 文件读写与异常处理。

```python
from pathlib import Path


def count_lines(path: str) -> int:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with file_path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)
```

这段代码同时体现了函数、类型标注、路径处理、异常和上下文管理器。学习时不要只问“语法是什么意思”，还要问“这个设计避免了什么错误”。

### 函数、模块与包

当代码超过几十行时，开始拆分职责。一个函数最好完成一个明确任务；相关函数放进模块；多个模块组成包。

```text
project/
├── pyproject.toml
├── README.md
├── src/
│   └── project/
│       ├── __init__.py
│       ├── data.py
│       └── model.py
└── tests/
    └── test_data.py
```

不要依赖“从某个目录运行才不报错”的偶然环境。理解当前工作目录、模块搜索路径和包导入关系。

### 面向对象编程

类适合表达“具有状态和行为的对象”，但不是所有代码都需要写成类。

```python
class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0
```

需要理解实例属性、方法、继承与组合。实践中优先考虑组合：让多个简单对象协作，通常比建立复杂继承层级更容易维护。

## 2.2 环境管理

每个项目使用独立虚拟环境，避免不同项目的依赖互相冲突：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

安装依赖后记录版本。团队项目可以使用 `requirements.txt`、`pyproject.toml` 或其他锁文件，但不要只把“我的电脑能运行”当作可复现。

你应能回答：

- 当前使用的是哪个 Python 解释器？
- 包安装到了哪个环境？
- 终端和 IDE 是否使用同一个环境？
- CUDA、PyTorch 与驱动版本是否匹配？

## 2.3 IDE 与调试

PyCharm、VS Code 或其他 IDE 都可以，关键是会使用：

- 选择解释器；
- 设置运行参数和环境变量；
- 断点、单步执行和变量查看；
- 跳转到定义与查找引用；
- 格式化、静态检查和测试运行。

遇到错误时按顺序处理：

1. 从最后一行识别错误类型；
2. 沿 traceback 找到第一处属于自己代码的位置；
3. 打印或检查输入值、类型和形状；
4. 构造最小复现；
5. 查官方文档和相关 issue；
6. 再向同学或 AI 工具提问。

提问时提供运行环境、完整报错、最小代码、预期结果和已经尝试的方法。

## 2.4 Git 基础

至少熟悉下面的工作流：

```bash
git status
git diff
git add path/to/file
git commit -m "Describe the change"
git pull --rebase
git push
```

提交应当小而完整：一次提交解决一个问题，并能通过基本测试。不要把数据集、模型权重、密钥、虚拟环境和临时输出提交到仓库。

## 2.5 Linux 与远程服务器

### 常用命令

需要熟悉文件与进程的基本操作：

```bash
pwd                 # 当前目录
ls -lah             # 查看文件
cd path             # 切换目录
mkdir -p outputs    # 创建目录
cp source target    # 复制文件
mv old new          # 移动或重命名
ps aux              # 查看进程
top                 # 查看资源占用
nvidia-smi          # 查看 GPU 状态
```

删除和覆盖前必须确认目标路径。不要复制不理解的递归删除命令。

### SSH 登录

```bash
ssh username@server.example.com
```

建议使用密钥认证，并在 `~/.ssh/config` 中为常用服务器建立别名：

```text
Host lab-server
    HostName server.example.com
    User username
    IdentityFile ~/.ssh/id_ed25519
```

之后可以直接运行 `ssh lab-server`。私钥不得上传到代码仓库或发送给他人。

### 上传与下载

```bash
scp local_file.py lab-server:~/project/
scp lab-server:~/project/result.json ./
rsync -av --progress ./project/ lab-server:~/project/
```

`scp` 适合少量文件，`rsync` 更适合重复同步项目。数据和模型较大时，先确认服务器存储位置和实验室规范。

### 长时间任务

SSH 断开后，前台进程通常会结束。可以使用 `tmux` 或任务调度系统：

```bash
tmux new -s experiment
python train.py
# Ctrl-b d 暂时离开
tmux attach -t experiment
```

如果服务器使用 Slurm，应学习 `sbatch`、资源申请和日志查看，而不是直接占用登录节点训练。

## 2.6 正确使用 GPT 等工具

AI 工具适合解释报错、补充测试、比较实现和查找学习路径，但你仍需验证结果：

- 要求它解释原因，而不只是给出可复制代码；
- 让它为代码补充边界测试；
- 对照官方文档检查 API；
- 不上传密钥、未公开数据和敏感研究内容；
- 运行前逐行理解涉及文件、网络和系统权限的命令。

## 本章任务

完成一个命令行文本统计工具：读取 UTF-8 文件，输出行数、字符数和出现频率最高的词。要求：

1. 使用函数和至少一个类；
2. 对空文件与不存在路径进行处理；
3. 编写三个自动测试；
4. 用 Git 管理代码；
5. 通过 SSH 上传到服务器并成功运行。

## 参考资源

- [Python 官方教程](https://docs.python.org/3/tutorial/)
- [Git 官方文档](https://git-scm.com/docs)
- [OpenSSH 手册](https://www.openssh.com/manual.html)
