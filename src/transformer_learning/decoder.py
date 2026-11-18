"""
========================================
decoder.py - Transformer 解码器
========================================

Decoder 和 Encoder 的区别：
    Encoder: 只有 Self-Attention
    Decoder: 有 Masked Self-Attention + Cross-Attention

每一层 Decoder 的结构：

    输入 x（已生成的 token）
      │
      ▼
    ┌──────────────────────────┐
    │   Masked Multi-Head       │  ← 因果掩码：只能看到之前生成的 token
    │   Self-Attention          │     Q = K = V = x
    └──────────┬───────────────┘
               │
      x + MaskedAttn(x)         ← 残差连接
               │
         LayerNorm
               │
      ┌────────┴────────┐
      │   Cross-Attention │  ← Q 来自 Decoder，K 和 V 来自 Encoder
      └────────┬────────┘      这就是 Decoder "读取" Encoder 信息的方式
               │
      x + CrossAttn(x)         ← 残差连接
               │
         LayerNorm
               │
      ┌────────┴────────┐
      │  Feed-Forward    │
      └────────┬────────┘
               │
      x + FFN(x)                ← 残差连接
               │
         LayerNorm
               │
            输出

为什么需要 Masked Self-Attention？
    在训练时，Decoder 的输入是完整的目标句子。
    但在推理时，模型是逐个生成 token 的——生成第 3 个词时不能看到第 4 个词。
    所以训练时也要用掩码模拟这个约束，保持训练和推理的一致性。

为什么需要 Cross-Attention？
    Decoder 需要"参考"Encoder 对源语言的编码结果。
    Cross-Attention 就是 Decoder 向 Encoder "提问"的机制。
"""

import torch
import torch.nn as nn

from transformer_learning.config import ModelArgs
from transformer_learning.multihead_attention import MultiHeadAttention
from transformer_learning.FFN import FeedForward


class DecoderLayer(nn.Module):
    """
    单层 Decoder = Masked Self-Attention + Cross-Attention + FFN。

    参数:
        args: ModelArgs 配置对象
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        # ---- 子层 1：带因果掩码的自注意力 ----
        # is_causal=True → 自动添加上三角掩码
        self.masked_self_attn = MultiHeadAttention(args, is_causal=True)

        # ---- 子层 2：交叉注意力 ----
        # Q 来自 Decoder，K 和 V 来自 Encoder
        self.cross_attn = MultiHeadAttention(args, is_causal=False)

        # ---- 子层 3：前馈网络 ----
        self.ffn = FeedForward(args)

        # 三个 LayerNorm
        self.norm1 = nn.LayerNorm(args.dim)
        self.norm2 = nn.LayerNorm(args.dim)
        self.norm3 = nn.LayerNorm(args.dim)

        self.dropout = nn.Dropout(args.dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        参数:
            x:              (B, T_tgt, dim)  — Decoder 的输入
            encoder_output: (B, T_src, dim)  — Encoder 的输出
            src_mask:       源序列的 padding mask
            tgt_mask:       目标序列的 padding mask

        返回:
            (B, T_tgt, dim)
        """
        # ---- 子层 1：Masked Self-Attention ----
        # Decoder 看自己已经生成的内容，但不能偷看未来
        attn_output = self.masked_self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + attn_output)

        # ---- 子层 2：Cross-Attention ----
        # Q 来自 Decoder，K 和 V 来自 Encoder
        # 这就是 Decoder "读取"源语言信息的方式
        cross_output = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
        x = self.norm2(x + cross_output)

        # ---- 子层 3：FFN ----
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x


class Decoder(nn.Module):
    """
    完整的 Decoder = N 层 DecoderLayer 堆叠。

    参数:
        args: ModelArgs 配置对象
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(args) for _ in range(args.n_layers)
        ])

        self.norm = nn.LayerNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        参数:
            x:              (B, T_tgt, dim)  — 嵌入后的目标序列
            encoder_output: (B, T_src, dim)  — Encoder 的输出
            src_mask:       源序列 padding mask
            tgt_mask:       目标序列 padding mask

        返回:
            (B, T_tgt, dim) — 解码后的序列
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


# ============ 简单测试 ============
if __name__ == "__main__":
    args = ModelArgs()
    decoder = Decoder(args)

    # Decoder 输入（目标序列嵌入）
    tgt = torch.randn(2, 8, 512)
    # Encoder 输出（源序列编码）
    enc_out = torch.randn(2, 10, 512)

    output = decoder(tgt, enc_out)

    print("Decoder 输入形状:", tgt.shape)      # (2, 8, 512)
    print("Encoder 输出形状:", enc_out.shape)    # (2, 10, 512)
    print("Decoder 输出形状:", output.shape)     # (2, 8, 512)
    print(f"Decoder 参数量: {sum(p.numel() for p in decoder.parameters()):,}")
