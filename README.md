# 🤖 Transformer Learning — 从零实现 Transformer

> 一个面向初学者的 Transformer 完整实现，代码注释详尽，适合学习论文
> ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)。

## 📁 项目结构

```
transformer-learning/
├── pyproject.toml                           # 项目配置 & 依赖
├── README.md                                # 你正在看的文件
└── src/
    └── transformer_learning/
        ├── __init__.py                      # 模块入口
        ├── config.py                        # ⚙️  超参数配置
        ├── attention.py                     # 🔍 缩放点积注意力
        ├── multihead_attention.py           # 🧠 多头注意力
        ├── FFN.py                           # 📦 前馈神经网络
        ├── embedding.py                     # 📝 词嵌入 + 位置编码
        ├── Encoder.py                       # 🏗️  编码器
        ├── decoder.py                       # 🏗️  解码器
        └── transformer.py                   # 🤖 完整 Transformer
```

## 🧩 模块说明

| 文件 | 对应论文部分 | 一句话总结 |
|------|-------------|-----------|
| `config.py` | — | 所有超参数集中管理 |
| `attention.py` | §3.2.1 | `Attention(Q,K,V) = softmax(QK^T/√d_k)V` |
| `multihead_attention.py` | §3.2.2 | 多个头并行计算注意力，捕捉不同模式 |
| `FFN.py` | §3.3 | 两层全连接 + ReLU，每个 token 独立处理 |
| `embedding.py` | §3.4 & §3.5 | 词嵌入 + 正弦/余弦位置编码 |
| `Encoder.py` | §3.1 | N 层 (Self-Attn + FFN + 残差 + LayerNorm) |
| `decoder.py` | §3.1 | N 层 (Masked Self-Attn + Cross-Attn + FFN) |
| `transformer.py` | 完整模型 | 组装所有组件，输入 token → 输出概率 |

## 🏗️ 架构图

```
源句子 (src)                     目标句子 (tgt)
    │                                │
    ▼                                ▼
┌──────────────┐              ┌──────────────┐
│  Token Embed  │              │  Token Embed  │
│  + Pos Enc    │              │  + Pos Enc    │
└──────┬───────┘              └──────┬───────┘
       │                             │
       ▼                             ▼
┌──────────────┐              ┌──────────────────┐
│   Encoder     │  ─────────▶ │     Decoder       │
│   × N 层      │   K, V      │     × N 层        │
└──────────────┘              └──────┬───────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │  Linear +     │
                              │  Softmax      │
                              └──────────────┘
                                     │
                                     ▼
                              预测的下一个词
```

## 🚀 快速开始

```bash
# 安装依赖
uv sync

# 运行演示
uv run transformer-learning

# 运行单个模块的测试
uv run python -m transformer_learning.attention
uv run python -m transformer_learning.multihead_attention
uv run python -m transformer_learning.FFN
uv run python -m transformer_learning.embedding
uv run python -m transformer_learning.Encoder
uv run python -m transformer_learning.decoder
uv run python -m transformer_learning.transformer
```

## 📖 学习建议

推荐按以下顺序阅读代码：

1. **`config.py`** — 先了解所有超参数
2. **`attention.py`** — Transformer 最核心的运算
3. **`multihead_attention.py`** — 理解"多头"的拆分与合并
4. **`FFN.py`** — 最简单的模块，两层全连接
5. **`embedding.py`** — 位置编码的数学原理
6. **`Encoder.py`** — 把注意力和 FFN 组装起来
7. **`decoder.py`** — 比 Encoder 多了交叉注意力
8. **`transformer.py`** — 最终的完整模型

## 📚 参考资料

- [Attention Is All You Need (原论文)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (图解)](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer (代码注解)](https://nlp.seas.harvard.edu/annotated-transformer/)
