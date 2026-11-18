"""
========================================
FFN.py - 前馈神经网络 (Position-wise Feed-Forward Network)
========================================

论文公式：
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

也就是两层全连接 + ReLU 激活：
    x → Linear(dim → d_ff) → ReLU → Dropout → Linear(d_ff → dim) → Dropout

直觉理解：
    注意力层负责 "信息交互"（token 之间互相看）。
    FFN 层负责 "信息加工"（每个 token 独立地进行非线性变换）。

    你可以把它想象成：
      注意力 = 开会讨论、交换信息
      FFN    = 回到工位、独立思考、整理笔记

    "Position-wise" 的意思是：每个位置（token）用同一套参数，
    但各自独立计算（不看其他 token）。

为什么中间层要扩大 4 倍？
    dim=512 → d_ff=2048 → dim=512
    这是论文的设定。先扩大再压缩，可以让模型在更高维空间里
    做非线性变换，增强表达能力（类似瓶颈结构的反面）。
"""

import torch
import torch.nn as nn

from transformer_learning.config import ModelArgs


class FeedForward(nn.Module):
    """
    位置逐点前馈网络。

    结构：Linear → ReLU → Dropout → Linear → Dropout

    参数:
        args: ModelArgs 配置对象
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        # 第一层：升维  dim → d_ff
        # 把每个 token 的 512 维表示投影到 2048 维
        self.w1 = nn.Linear(args.dim, args.d_ff)

        # 第二层：降维  d_ff → dim
        # 再把 2048 维压缩回 512 维
        self.w2 = nn.Linear(args.d_ff, args.dim)

        # ReLU 激活函数
        # max(0, x)：小于 0 的值全部变成 0，大于 0 的值保持不变
        # 引入非线性，让模型能学到复杂的映射
        self.relu = nn.ReLU()

        # Dropout：训练时随机丢弃一部分神经元，防止过拟合
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: (B, T, dim)  — 每个 token 的表示

        返回:
            output: (B, T, dim) — 维度不变！
        """
        # x: (B, T, 512)
        x = self.w1(x)        # → (B, T, 2048)   升维
        x = self.relu(x)      # → (B, T, 2048)   非线性激活
        x = self.dropout(x)   # → (B, T, 2048)   随机丢弃
        x = self.w2(x)        # → (B, T, 512)    降维
        x = self.dropout(x)   # → (B, T, 512)    再次随机丢弃
        return x


# ============ 简单测试 ============
if __name__ == "__main__":
    args = ModelArgs()
    ffn = FeedForward(args)

    x = torch.randn(2, 10, 512)
    output = ffn(x)

    print("输入形状:", x.shape)    # (2, 10, 512)
    print("输出形状:", output.shape) # (2, 10, 512)
    print(f"参数量: {sum(p.numel() for p in ffn.parameters()):,}")
