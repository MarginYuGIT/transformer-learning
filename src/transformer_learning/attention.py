"""
========================================
attention.py - 缩放点积注意力 (Scaled Dot-Product Attention)
========================================

这是 Transformer 最核心的运算，论文公式：

    Attention(Q, K, V) = softmax( QK^T / √d_k ) · V

直觉理解：
  - Q (Query)  = "我在找什么"
  - K (Key)    = "我有什么标签"
  - V (Value)  = "我实际的内容"

步骤：
  1. Q 和 K 做点积 → 得到"相关性分数"
  2. 除以 √d_k → 防止分数太大导致 softmax 梯度消失
  3. （可选）加掩码 → 把不该看的位置设为 -∞
  4. softmax → 把分数变成概率（权重之和 = 1）
  5. 用权重对 V 加权求和 → 得到输出
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    缩放点积注意力。

    参数:
        query:   (batch, heads, seq_q, d_k)   —— 查询向量
        key:     (batch, heads, seq_k, d_k)   —— 键向量
        value:   (batch, heads, seq_k, d_v)   —— 值向量
        mask:    (1, 1, seq_q, seq_k) 或兼容形状  —— 掩码张量
        dropout: nn.Dropout 实例

    返回:
        output:  (batch, heads, seq_q, d_v)   —— 注意力输出
        attn:    (batch, heads, seq_q, seq_k) —— 注意力权重（方便可视化）
    """

    # ---- 第 1 步：获取 d_k，也就是每个 Query/Key 向量的维度 ----
    d_k = query.size(-1)

    # ---- 第 2 步：计算相似度分数 ----
    # Q @ K^T → (batch, heads, seq_q, seq_k)
    # 除以 √d_k 是为了让方差稳定在 1 左右，防止 softmax "饱和"
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # ---- 第 3 步：施加掩码 ----
    # 被掩码的位置填入一个极小值（-1e9），这样 softmax 后权重趋近于 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # ---- 第 4 步：Softmax 归一化 ----
    # dim=-1 表示对最后一个维度（seq_k 维度）做归一化
    # 也就是：对于每个 query，它对所有 key 的注意力权重之和 = 1
    attn = F.softmax(scores, dim=-1)

    # ---- 第 5 步：Dropout（训练时随机丢弃一些注意力连接，防止过拟合）----
    if dropout is not None:
        attn = dropout(attn)

    # ---- 第 6 步：用权重对 Value 加权求和 ----
    # attn @ V → (batch, heads, seq_q, d_v)
    output = torch.matmul(attn, value)

    return output, attn


# ============ 简单测试 ============
if __name__ == "__main__":
    # 创建测试张量：batch=2, heads=8, seq_len=5, d_k=64
    Q = torch.randn(2, 8, 5, 64)
    K = torch.randn(2, 8, 5, 64)
    V = torch.randn(2, 8, 5, 64)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("输出形状:", output.shape)     # (2, 8, 5, 64)
    print("注意力权重形状:", weights.shape) # (2, 8, 5, 5)
    # 验证权重每行之和 ≈ 1
    print("权重行和:", weights[0, 0, 0].sum().item())
