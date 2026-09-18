# Transformer from Scratch

这是一个使用 **PyTorch 从零实现 Transformer Encoder-Decoder 架构** 的学习项目。

项目没有直接使用 `torch.nn.Transformer`，而是手动实现了 Transformer 的核心模块，包括：

- Token Embedding
- Positional Encoding
- Multi-Head Attention
- Encoder / Decoder
- Cross Attention
- Feed Forward Network
- Residual Connection
- Layer Normalization
- Padding Mask
- Causal Mask
- Greedy Decoding

当前使用 **Helsinki-NLP / OPUS Books** 英意翻译数据集进行训练。

---

## 1. 项目结构

```text
Transformer/
├── config.py
├── dataset.py
├── model.py
├── train.py
├── tokenizers/
├── weights/
├── runs/
├── .gitignore
└── README.md
```

其中：

- `model.py`：实现 Transformer 模型结构
- `dataset.py`：处理双语数据、Mask 和训练样本
- `train.py`：训练、验证、Greedy Decode、日志记录
- `config.py`：模型和训练参数配置
- `tokenizers/`：保存训练后的 tokenizer
- `weights/`：保存模型 checkpoint
- `runs/`：保存 TensorBoard 日志

---

## 2. 模型结构

整体结构如下：

```text
Source Sentence
      |
      v
Token Embedding
      +
Positional Encoding
      |
      v
+----------------------+
|       Encoder        |
|                      |
|  Self-Attention      |
|        ↓             |
|  Feed Forward        |
+----------------------+
      |
      | Encoder Output
      |
      v
+----------------------+
|       Decoder        |
|                      |
| Masked Self-Attention|
|        ↓             |
| Cross Attention      |
|        ↓             |
| Feed Forward         |
+----------------------+
      |
      v
Linear Projection
      |
      v
Target Vocabulary
```

---

## 3. 已实现模块

### Input Embedding

将 token id 映射为 `d_model` 维向量：

```text
Token ID
   ↓
Embedding
   ↓
d_model 维向量
```

并按照原论文对 embedding 进行缩放：

```text
Embedding × sqrt(d_model)
```

---

### Positional Encoding

Transformer 本身没有 RNN 的顺序结构，因此需要显式加入位置信息。

使用原论文中的正弦位置编码：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))

PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

最终输入：

```text
Token Embedding
      +
Positional Encoding
```

---

### Multi-Head Attention

首先计算：

```text
Q = XWq
K = XWk
V = XWv
```

Scaled Dot-Product Attention：

```text
Attention(Q, K, V)
=
softmax(QK^T / sqrt(d_k)) V
```

多个 Attention Head 分别计算：

```text
head1
head2
head3
...
headh
```

然后：

```text
Concat(head1, ..., headh)
        ↓
Linear Projection
```

---

## 4. Encoder

每个 Encoder Block 主要包含：

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
Feed Forward Network
  ↓
Residual Connection
```

多个 Encoder Block 堆叠形成完整 Encoder。

---

## 5. Decoder

Decoder Block 包含三部分：

```text
Masked Self-Attention
        ↓
Cross Attention
        ↓
Feed Forward Network
```

其中 Cross Attention 中：

```text
Query
=
Decoder Hidden State
```

而：

```text
Key / Value
=
Encoder Output
```

因此 Decoder 可以在生成目标语言 token 时读取源语言信息。

---

## 6. Causal Mask

Decoder 在训练过程中不能看到未来 token。

例如序列长度为 4 时：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

这意味着：

```text
位置 1 只能看到位置 1

位置 2 可以看到位置 1、2

位置 3 可以看到位置 1、2、3
```

从而防止 Decoder 在训练时提前看到未来答案。

---

## 7. 数据处理

当前使用数据集：

```text
Helsinki-NLP/opus_books
```

语言配置：

```text
en-it
```

即：

```text
English → Italian
```

例如：

```text
SOURCE:
She blushed and stopped.

TARGET:
Arrossì e si fermò.
```

---

## 8. Tokenizer

使用 Hugging Face `tokenizers` 实现 WordLevel tokenizer。

特殊 token：

```text
[UNK]
[PAD]
[SOS]
[EOS]
```

Tokenizer：

```python
Tokenizer(
    WordLevel(
        unk_token="[UNK]"
    )
)
```

并使用：

```python
Whitespace()
```

进行预分词。

---

## 9. Encoder / Decoder 输入

Encoder Input：

```text
[SOS] + source tokens + [EOS] + [PAD] ...
```

Decoder Input：

```text
[SOS] + target tokens + [PAD] ...
```

Label：

```text
target tokens + [EOS] + [PAD] ...
```

因此 Decoder 实际上是在学习：

```text
输入：
[SOS] I love

预测：
I love you
```

这种自回归生成过程。

---

## 10. Decoder Mask

Decoder Mask 由两部分组成：

```text
Padding Mask
+
Causal Mask
```

Padding Mask 用于忽略：

```text
[PAD]
```

Causal Mask 用于禁止模型看到未来 token。

---

## 11. 训练流程

整体训练流程：

```text
Source Tokens
      |
      v
Embedding
      |
      v
Positional Encoding
      |
      v
Encoder
      |
      v
Encoder Output
      |
      +----------------------+
                             |
Target Tokens                |
      |                      |
      v                      |
Embedding                    |
      |                      |
      v                      |
Positional Encoding          |
      |                      |
      v                      |
Masked Self-Attention        |
      |                      |
      v                      |
Cross Attention <------------+
      |
      v
Feed Forward
      |
      v
Projection Layer
      |
      v
Vocabulary Logits
      |
      v
Cross Entropy Loss
```

---

## 12. Loss

训练使用：

```python
nn.CrossEntropyLoss(
    ignore_index=pad_token_id,
    label_smoothing=0.1
)
```

其中：

```text
ignore_index
```

用于忽略 `[PAD]` token。

同时使用：

```text
label_smoothing = 0.1
```

降低模型对 one-hot 标签的过拟合。

---

## 13. Greedy Decoding

验证阶段使用 Greedy Decode。

从：

```text
[SOS]
```

开始。

每一步选择当前概率最大的 token：

```text
[SOS]
   ↓
token1
   ↓
[SOS] token1
   ↓
token2
   ↓
[SOS] token1 token2
   ↓
...
   ↓
[EOS]
```

当模型生成 `[EOS]` 或达到最大长度后停止。

---

## 14. 安装依赖

```bash
pip install torch
pip install datasets
pip install tokenizers
pip install tqdm
pip install tensorboard
pip install torchmetrics
```

也可以：

```bash
pip install torch datasets tokenizers tqdm tensorboard torchmetrics
```

---

## 15. 开始训练

运行：

```bash
python train.py
```

如果需要同时保存终端日志：

```bash
python -u train.py 2>&1 | tee run.log
```

训练过程类似：

```text
Processing Epoch 00:
3638/3638 [07:10, 8.46it/s, loss=5.988]
```

---

## 16. TensorBoard

项目使用 TensorBoard 记录训练指标。

启动：

```bash
tensorboard --logdir runs
```

然后浏览器访问：

```text
http://localhost:6006
```

即可查看：

```text
Train Loss
Validation Metrics
```

等训练曲线。

---

## 17. 模型保存

模型 checkpoint 保存在：

```text
weights/
```

每个 checkpoint 包含：

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "global_step": global_step
}
```

可以用于继续训练或后续推理。

---

## 18. 项目目标

这个项目主要用于理解 Transformer 的底层实现，而不是构建生产级机器翻译系统。

希望通过手动实现理解：

- Token Embedding 是如何工作的
- Transformer 为什么需要位置编码
- Q、K、V 是如何得到的
- Attention Score 是如何计算的
- Multi-Head Attention 为什么要分多个 Head
- Encoder 和 Decoder 有什么区别
- Cross Attention 的作用是什么
- Padding Mask 和 Causal Mask 有什么区别
- Decoder 为什么要进行自回归生成
- Transformer 如何完成机器翻译任务

---

## 19. 参考论文

Vaswani et al.

**Attention Is All You Need**

NeurIPS 2017

论文：

https://arxiv.org/abs/1706.03762

---

## 说明

本仓库主要用于 Transformer 原理学习和代码实践。

模型结构以原始 Encoder-Decoder Transformer 为主，不使用 `torch.nn.Transformer` 封装模块，而是手动实现各个核心组件。
