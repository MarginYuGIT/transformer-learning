"""
========================================
embedding.py - 词嵌入 + 位置编码 (Embedding + Positional Encoding)
========================================

Transformer 的输入需要两个信息：
    1. 这个 token 是什么词？ → 词嵌入 (Token Embedding)
    2. 这个 token 在什么位置？ → 位置编码 (Positional Encoding)

最终输入 = 词嵌入 + 位置编码

为什么需要位置编码？
    注意力机制是"无序"的 —— 它只看内容相关性，不知道词的先后顺序。
    但语言是有顺序的："猫吃鱼" ≠ "鱼吃猫"。
    所以我们需要手动注入位置信息。

位置编码使用正弦/余弦函数（论文原始方案）：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

为什么用 sin/cos？
    - 能表示任意长度的序列（外推能力）
    - 不同位置的编码是唯一的
    - 相对位置可以通过线性变换得到
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    词嵌入层：把离散的 token ID 映射成连续的向量。

    举例：
        词表大小 = 10000, 嵌入维度 = 512
        输入: [42, 100, 7]  (3 个 token ID)
        输出: 3 × 512 的矩阵（每个 ID 对应一行 512 维向量）

    这些向量是可学习的参数，训练过程中会不断更新。
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, T) — token ID 序列

        返回:
            (B, T, d_model) — 嵌入向量

        注意：乘以 √d_model 是论文的做法，
        目的是让嵌入值的量级和位置编码匹配。
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码。

    这个模块不包含可学习的参数，纯粹是数学公式计算出来的固定值。
    通过 register_buffer 注册，这样它会随模型一起保存和移动到 GPU。

    编码矩阵长这样（以 d_model=4 为例）：
        位置0: [sin(0), cos(0), sin(0), cos(0)]
        位置1: [sin(1/1), cos(1/1), sin(1/10000), cos(1/10000)]
        位置2: [sin(2/1), cos(2/1), sin(2/10000), cos(2/10000)]
        ...
    低维度变化快（高频），高维度变化慢（低频），就像时钟的秒针和时针。
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # ---- 创建位置编码矩阵 ----
        pe = torch.zeros(max_seq_len, d_model)  # (max_seq_len, d_model)

        # position: 每个位置的索引 [0, 1, 2, ..., max_seq_len-1]
        # shape: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # div_term: 分母中的 10000^(2i/d_model)
        # 这里用 exp + log 来计算，数值上更稳定
        # 2i 的范围是 [0, 2, 4, ..., d_model-2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # 1, 3, 5, ...

        # 增加 batch 维度：(max_seq_len, d_model) → (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer: 不是模型参数（不参与梯度更新），
        # 但会随模型一起 save/load 和 to(device)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        给输入加上位置编码。

        参数:
            x: (B, T, d_model) — 词嵌入向量

        返回:
            (B, T, d_model) — 加上位置信息后的向量
        """
        # 只取前 T 个位置的编码（T ≤ max_seq_len）
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    完整的嵌入层 = 词嵌入 + 位置编码。

    这是 Transformer 的"入口"，把离散的 token ID 转化为
    包含内容信息和位置信息的连续向量。

    流程图：
        token IDs → [TokenEmbedding] → 词向量 → [+ PositionalEncoding] → 最终嵌入
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, T) — token ID 序列

        返回:
            (B, T, d_model) — 嵌入 + 位置编码
        """
        return self.pos_enc(self.token_emb(x))


# ============ 简单测试 ============
if __name__ == "__main__":
    # 模拟输入：batch=2, seq_len=10, 词表大小=1000, 嵌入维度=512
    vocab_size = 1000
    d_model = 512
    max_seq_len = 100

    emb = TransformerEmbedding(vocab_size, d_model, max_seq_len, dropout=0.1)

    # 随机生成 token ID（范围 0 ~ 999）
    token_ids = torch.randint(0, vocab_size, (2, 10))

    output = emb(token_ids)
    print("Token IDs 形状:", token_ids.shape)  # (2, 10)
    print("嵌入输出形状:", output.shape)         # (2, 10, 512)
