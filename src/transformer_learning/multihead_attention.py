"""
========================================
multihead_attention.py - 多头注意力 (Multi-Head Attention)
========================================

为什么需要"多头"？
  单头注意力只能学到一种"关注模式"。
  多头注意力让模型同时学到多种不同的关注模式：
    - 有的头关注语法关系（主谓宾）
    - 有的头关注语义相似性
    - 有的头关注位置距离

论文公式：
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    其中 head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

实现步骤（本文件的做法）：
    1. 用一个大的线性层一次性投影出所有头的 Q/K/V
    2. reshape + transpose 把 (B, T, D) 变成 (B, H, T, d)
    3. 对每个头独立做缩放点积注意力
    4. transpose + reshape 合并回 (B, T, D)
    5. 过最后一个线性层 W^O
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_learning.config import ModelArgs
from transformer_learning.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制。

    构造参数:
        args:      ModelArgs 配置对象
        is_causal: 是否使用因果掩码（Decoder 自注意力时设为 True）
    """

    def __init__(self, args: ModelArgs, is_causal: bool = False):
        super().__init__()

        # ---- 检查：模型维度必须能被头数整除 ----
        # 例如 dim=512, n_heads=8 → 每个头的维度 head_dim = 64 ✓
        # 如果 dim=512, n_heads=7 → 512/7 = 73.14… ✗ 不行！
        assert args.dim % args.n_heads == 0, \
            f"dim({args.dim}) 必须能被 n_heads({args.n_heads}) 整除"

        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads  # 每个头的维度
        self.is_causal = is_causal

        # ---- Q、K、V 的线性投影 ----
        # 把输入 X 分别映射到 Q、K、V 空间
        # 公式：Q = X @ W_q,  K = X @ W_k,  V = X @ W_v
        #
        # 为什么 forward 要分开传 q, k, v 而不是传一个 x？
        # → 为了支持"交叉注意力"（Cross-Attention）：
        #   在 Decoder 中，Q 来自 Decoder，K 和 V 来自 Encoder
        self.wq = nn.Linear(args.n_embd, args.dim, bias=False)
        self.wk = nn.Linear(args.n_embd, args.dim, bias=False)
        self.wv = nn.Linear(args.n_embd, args.dim, bias=False)

        # ---- 多头拼接后的输出投影 W^O ----
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        # ---- Dropout ----
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        # ---- 因果掩码（Causal Mask）----
        # 用于 Decoder 的自注意力：禁止当前 token 看到未来的 token
        # 掩码形状示例（seq_len=4）：
        #   [[0, -∞, -∞, -∞],
        #    [0,  0, -∞, -∞],
        #    [0,  0,  0, -∞],
        #    [0,  0,  0,  0]]
        # 加到 scores 上后，-∞ 位置经过 softmax 变成 0
        if is_causal:
            mask = torch.full(
                (args.max_seq_len, args.max_seq_len), float("-inf")
            )
            # torch.triu: 取上三角矩阵，diagonal=1 表示主对角线上方
            self.register_buffer("mask", torch.triu(mask, diagonal=1))

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 (B, T, D) 重塑为 (B, H, T, d)。

        举例：dim=512, n_heads=8
          输入: (2, 10, 512)
          → view: (2, 10, 8, 64)
          → transpose(1,2): (2, 8, 10, 64)

        为什么要这样做？
          我们希望对每个头的所有 token 做注意力计算，
          所以要让 "头" 的维度放到前面（方便矩阵乘法并行计算）
        """
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, H, T, d)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        split_heads 的逆操作：把 (B, H, T, d) 合并回 (B, T, D)。

        举例：
          输入: (2, 8, 10, 64)
          → transpose(1,2): (2, 10, 8, 64)
          → reshape: (2, 10, 512)
        """
        B, H, T, d = x.shape
        x = x.transpose(1, 2).contiguous()  # (B, T, H, d)
        return x.reshape(B, T, H * d)       # (B, T, D)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向传播。

        参数:
            q: (B, T_q, n_embd)  — 查询来源（Self-Attn 时和 k, v 相同）
            k: (B, T_k, n_embd)  — 键来源
            v: (B, T_k, n_embd)  — 值来源
            mask: 可选掩码（padding mask 或因果掩码外部传入都可以）

        返回:
            output: (B, T_q, dim)
        """

        # ---- 步骤 1：线性投影 + 拆分多头 ----
        # nn.Linear 只对最后一个维度做变换，前面的 B, T 维度不受影响
        Q = self.split_heads(self.wq(q))  # (B, H, T_q, d)
        K = self.split_heads(self.wk(k))  # (B, H, T_k, d)
        V = self.split_heads(self.wv(v))  # (B, H, T_k, d)

        # ---- 步骤 2：准备掩码 ----
        if self.is_causal:
            T = Q.size(-2)
            # 因果掩码：只取前 T×T 的部分
            causal_mask = self.mask[:T, :T]  # (T, T)
            # 转换为 0/1 掩码供 attention 函数使用
            # mask==0 的位置表示"可以关注"
            attn_mask = (causal_mask == 0).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            if mask is not None:
                attn_mask = attn_mask & mask  # 同时应用 padding mask
        else:
            attn_mask = mask

        # ---- 步骤 3：缩放点积注意力 ----
        output, _ = scaled_dot_product_attention(
            Q, K, V, mask=attn_mask, dropout=self.attn_dropout
        )

        # ---- 步骤 4：合并多头 ----
        output = self.merge_heads(output)  # (B, T_q, D)

        # ---- 步骤 5：输出投影 + Dropout ----
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output


# ============ 简单测试 ============
if __name__ == "__main__":
    args = ModelArgs()
    model = MultiHeadAttention(args, is_causal=True)

    x = torch.randn(2, 5, 512)
    output = model(x, x, x)  # Self-Attention: Q=K=V=x

    print("输入形状:", x.shape)   # (2, 5, 512)
    print("输出形状:", output.shape)  # (2, 5, 512)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
