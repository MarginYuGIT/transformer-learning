import torch
import math

def attention(query, key, value, dropout=None):
    # query: (batch, heads, n_q, d_k)
    # key:   (batch, heads, n_k, d_k)
    # value: (batch, heads, n_k, d_v)

    # 1. 获取每个 Query 向量的特征维度
    d_k = query.size(-1)

    # 2. 计算 Query 与 Key 的相似度
    scores = torch.matmul(
        query, key.transpose(-2, -1)
    ) / math.sqrt(d_k)

    # scores: (batch, heads, n_q, n_k)

    # 3. 对每个 Query 的分数进行归一化
    p_attn = scores.softmax(dim=-1)#dim = -1 按行进行softmax dim = 0,按列进行softmax

    # 4. 可选的 Dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 5. 用注意力权重对 Value 加权求和
    output = torch.matmul(p_attn, value)

    return output, p_attn


# 创建测试张量

Q = torch.randn(2, 8, 5, 64)
K = torch.randn(2, 8, 5, 64)
V = torch.randn(2, 8, 5, 32)

output, weights = attention(Q, K, V)

print("输出形状:", output.shape)
print("注意力权重形状:", weights.shape)