"""
========================================
Encoder.py - Transformer 编码器
========================================

Encoder 的结构（每一层）：

    输入 x
      │
      ▼
    ┌──────────────────────┐
    │   Multi-Head          │
    │   Self-Attention      │  ← Q = K = V = x（自注意力：自己看自己）
    └──────────┬───────────┘
               │
      x + Attention(x)      ← 残差连接 (Residual Connection)
               │
         LayerNorm           ← 层归一化
               │
      ┌────────┴────────┐
      │  Feed-Forward    │  ← 两层全连接 + ReLU
      └────────┬────────┘
               │
      x + FFN(x)            ← 残差连接
               │
         LayerNorm           ← 层归一化
               │
            输出

    这个结构重复 N 次（N = n_layers），就是完整的 Encoder。

残差连接 (Residual Connection) 的作用：
    output = x + SubLayer(x)
    - 解决深层网络的梯度消失问题
    - 让信息可以"跳过"某一层直接传递
    - 直觉：即使某一层学得不好，至少不会让结果变差

LayerNorm 的作用：
    - 对每个样本的特征维度进行归一化
    - 让训练更稳定、收敛更快
"""

import torch
import torch.nn as nn

from transformer_learning.config import ModelArgs
from transformer_learning.multihead_attention import MultiHeadAttention
from transformer_learning.FFN import FeedForward


class EncoderLayer(nn.Module):
    """
    单层 Encoder = Self-Attention + FFN + 残差连接 + LayerNorm。

    参数:
        args: ModelArgs 配置对象
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        # 多头自注意力（Encoder 不需要因果掩码，因为可以看到全部输入）
        self.self_attn = MultiHeadAttention(args, is_causal=False)

        # 前馈神经网络
        self.ffn = FeedForward(args)

        # 两个 LayerNorm（分别用于注意力子层和 FFN 子层之后）
        self.norm1 = nn.LayerNorm(args.dim)
        self.norm2 = nn.LayerNorm(args.dim)

        # Dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            x:        (B, T, dim) — 输入序列
            src_mask: 可选的 padding mask

        返回:
            (B, T, dim) — 输出序列（维度不变）
        """
        # ---- 子层 1：多头自注意力 + 残差 + LayerNorm ----
        # Self-Attention: Q = K = V = x
        attn_output = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + attn_output)  # 残差连接 + LayerNorm

        # ---- 子层 2：前馈网络 + 残差 + LayerNorm ----
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)   # 残差连接 + LayerNorm

        return x


class Encoder(nn.Module):
    """
    完整的 Encoder = N 层 EncoderLayer 堆叠。

    参数:
        args: ModelArgs 配置对象
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        # 用 ModuleList 创建 N 层 EncoderLayer
        # ModuleList 比普通 list 好在：PyTorch 能正确追踪其中的参数
        self.layers = nn.ModuleList([
            EncoderLayer(args) for _ in range(args.n_layers)
        ])

        # 最后再做一次 LayerNorm（有些实现会加这一步）
        self.norm = nn.LayerNorm(args.dim)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            x:        (B, T, dim) — 嵌入后的输入序列
            src_mask: 可选的 padding mask

        返回:
            (B, T, dim) — 编码后的序列
        """
        # 依次通过每一层
        for layer in self.layers:
            x = layer(x, src_mask)

        # 最终归一化
        return self.norm(x)


# ============ 简单测试 ============
if __name__ == "__main__":
    args = ModelArgs()
    encoder = Encoder(args)

    # 模拟嵌入后的输入：batch=2, seq_len=10, dim=512
    x = torch.randn(2, 10, 512)
    output = encoder(x)

    print("输入形状:", x.shape)    # (2, 10, 512)
    print("输出形状:", output.shape) # (2, 10, 512)
    print(f"Encoder 参数量: {sum(p.numel() for p in encoder.parameters()):,}")
