
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelArgs:
    n_embd = 512
    dim = 512
    n_heads = 8
    dropout = 0.1
    max_seq_len = 128


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, args: ModelArgs, is_causal=False):
        super().__init__()

        assert args.dim % args.n_heads == 0

        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.is_causal = is_causal

        # Q、K、V 的线性投影 也就是 Q = XWq
        self.wq = nn.Linear(args.n_embd, args.dim, bias=False)
        self.wk = nn.Linear(args.n_embd, args.dim, bias=False)
        self.wv = nn.Linear(args.n_embd, args.dim, bias=False)

        # 多头拼接后的输出投影
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        # 因果掩码：禁止关注未来 token
        if is_causal:
            mask = torch.full(
                (args.max_seq_len, args.max_seq_len),
                float("-inf")
            )
            self.register_buffer(
                "mask", torch.triu(mask, diagonal=1)
            )
    """
    我们希望对每个头的所有 token 进行注意力计算，而不是对每个 token 的所有头分别计算
    回顾self-attention:attention = softmax(QK^T)V
    多头注意力 multihead_attention = contact(head_1,2,3..i)
    where head_i = attention(Q_iK^T_i)V 自注意力机制就是一个头对所有的token做注意力计算
    """
    def split_heads(self, x):
        """(B, T, D) -> (B, H, T, d)"""
        B, T, _ = x.shape

        x = x.view(B, T, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, q, k, v):
        # 1. 线性投影并拆分多头
        """
        1 为什么不直接传入x再经过三个卷积层得到QKV，按照公式里来说Q,k,V=xW_q,k,v

          是因为为了实现交叉注意力，即Q,K,V来自不同的输入，比如带有encoder和decoder的transformer里V来自decoder而QK来自decoder

        2 为什么XW_q得以实现，X型状是[BTD],w_q函数是从输入embd到dim，这只是对一个维度做了处理为什么能适用于X这三个维度的

          因为nn.Linear 默认只对输入张量的最后一个维度进行线性变换，前面的所有维度保持不变

        """
        Q = self.split_heads(self.wq(q))
        K = self.split_heads(self.wk(k))
        V = self.split_heads(self.wv(v))
        
        """
        PyTorch 的 matmul 默认使用最后两个维度进行矩阵乘法。
        """
        # 2. 计算注意力分数
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # 3. 因果掩码
        if self.is_causal:
            T = Q.size(-2)
            assert K.size(-2) == T
            assert T <= self.mask.size(-1)
            scores = scores + self.mask[:T, :T]

        # 4. Softmax 计算注意力权重
        attn = F.softmax(scores.float(), dim=-1)
        attn = attn.to(Q.dtype)
        attn = self.attn_dropout(attn) # 为了防止模型过度依赖某几个固定的注意力连接，从而减少过拟合。

        # 5. 加权求和
        output = attn @ V

        # 6. 合并多头
        B, H, T, d = output.shape
        output = output.transpose(1, 2)
        output = output.reshape(B, T, H * d)

        # 7. 输出投影
        output = self.wo(output)
        return self.resid_dropout(output) # 负责融合多头结果


if __name__ == "__main__":
    args = ModelArgs()
    model = MultiHeadAttention(args, is_causal=True)

    x = torch.randn(2, 5, 512)
    output = model(x, x, x)

    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
