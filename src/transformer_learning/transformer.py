"""
========================================
transformer.py - 完整的 Transformer 模型
========================================

这是把所有组件组装起来的"主文件"。

完整的 Transformer 结构（用于序列到序列任务，如机器翻译）：

    源句子 (src)              目标句子 (tgt)
        │                         │
        ▼                         ▼
    ┌────────────┐           ┌────────────┐
    │  Embedding  │           │  Embedding  │
    │  + PosEnc   │           │  + PosEnc   │
    └─────┬──────┘           └─────┬──────┘
          │                        │
          ▼                        ▼
    ┌────────────┐           ┌────────────────┐
    │            │           │                │
    │  Encoder   │──────────▶│    Decoder     │
    │  × N 层    │  K, V     │    × N 层      │
    │            │           │                │
    └─────┬──────┘           └─────┬──────────┘
          │                        │
          │                        ▼
          │                  ┌────────────┐
          │                  │   Linear    │  ← 投影到词表大小
          │                  │  (Output)   │
          │                  └─────┬──────┘
          │                        │
          │                        ▼
          │                  ┌────────────┐
          │                  │  Softmax    │  ← 得到每个词的概率
          │                  └────────────┘
"""

import torch
import torch.nn as nn

from transformer_learning.config import ModelArgs
from transformer_learning.embedding import TransformerEmbedding
from transformer_learning.Encoder import Encoder
from transformer_learning.decoder import Decoder


class Transformer(nn.Module):
    """
    完整的 Transformer 模型（Encoder-Decoder 架构）。

    适用于：机器翻译、文本摘要、对话生成等序列到序列任务。

    参数:
        args: ModelArgs 配置对象
        src_vocab_size: 源语言词表大小（如果不同于 args.vocab_size）
        tgt_vocab_size: 目标语言词表大小（如果不同于 args.vocab_size）
    """

    def __init__(
        self,
        args: ModelArgs,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
    ):
        super().__init__()

        src_vocab = src_vocab_size or args.vocab_size
        tgt_vocab = tgt_vocab_size or args.vocab_size

        # ---- 嵌入层 ----
        # 源语言和目标语言各有自己的嵌入层
        self.src_embedding = TransformerEmbedding(
            src_vocab, args.dim, args.max_seq_len, args.dropout
        )
        self.tgt_embedding = TransformerEmbedding(
            tgt_vocab, args.dim, args.max_seq_len, args.dropout
        )

        # ---- 编码器和解码器 ----
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        # ---- 输出投影层 ----
        # 把 Decoder 的输出（dim 维）映射到目标语言词表大小
        # 然后就可以用 softmax 得到每个词的概率
        self.output_proj = nn.Linear(args.dim, tgt_vocab, bias=False)

        # ---- 初始化参数 ----
        self._init_parameters()

        # 保存配置
        self.args = args

    def _init_parameters(self):
        """
        Xavier 均匀初始化。

        Xavier 初始化的目的：让每一层的输入和输出方差保持一致，
        防止信号在前向传播时逐层放大或缩小。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, src: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        编码器前向传播（编码源序列）。

        参数:
            src:      (B, T_src)  — 源语言 token ID 序列
            src_mask: padding mask

        返回:
            (B, T_src, dim)  — 编码后的表示
        """
        src_emb = self.src_embedding(src)   # (B, T_src, dim)
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        解码器前向传播。

        参数:
            tgt:            (B, T_tgt)   — 目标语言 token ID 序列
            encoder_output: (B, T_src, dim) — 编码器输出
            src_mask:       源序列 padding mask
            tgt_mask:       目标序列 padding mask

        返回:
            (B, T_tgt, dim)  — 解码后的表示
        """
        tgt_emb = self.tgt_embedding(tgt)   # (B, T_tgt, dim)
        return self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        完整的前向传播：src → Encoder → Decoder → 输出 logits。

        参数:
            src:      (B, T_src)  — 源语言 token ID
            tgt:      (B, T_tgt)  — 目标语言 token ID
            src_mask: 源序列 padding mask
            tgt_mask: 目标序列 padding mask

        返回:
            logits: (B, T_tgt, tgt_vocab_size) — 每个位置对词表的分数
            （取 argmax 或 softmax 就能得到预测的词）
        """
        # 步骤 1: 编码源序列
        encoder_output = self.encode(src, src_mask)

        # 步骤 2: 解码目标序列（参考编码器输出）
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # 步骤 3: 投影到词表
        logits = self.output_proj(decoder_output)  # (B, T_tgt, vocab_size)

        return logits

    @staticmethod
    def make_pad_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        创建 padding mask。

        把 pad token 的位置标记为 False，非 pad 位置标记为 True。
        这样在注意力计算时，模型就不会关注 padding 位置。

        举例（pad_idx=0,也就是说将pad 的地方设置成0）：
            输入: [5, 3, 7, 0, 0]
            掩码: [True, True, True, False, False]

        参数:
            seq:     (B, T) — token ID 序列
            pad_idx: padding token 的 ID

        返回:
            (B, 1, 1, T) — 广播友好的掩码形状
        """
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


# ============ 简单测试 ============
if __name__ == "__main__":
    args = ModelArgs()
    args.vocab_size = 1000  # 用小词表方便测试

    model = Transformer(args)

    # 模拟输入
    src = torch.randint(1, 1000, (2, 10))  # 源句子：batch=2, len=10
    tgt = torch.randint(1, 1000, (2, 8))   # 目标句子：batch=2, len=8

    # 创建 padding mask
    src_mask = Transformer.make_pad_mask(src)
    tgt_mask = Transformer.make_pad_mask(tgt)

    # 前向传播
    logits = model(src, tgt, src_mask, tgt_mask)

    print("=" * 50)
    print("Transformer 模型测试")
    print("=" * 50)
    print(f"源句子形状:   {src.shape}")      # (2, 10)
    print(f"目标句子形状: {tgt.shape}")       # (2, 8)
    print(f"输出 logits:  {logits.shape}")    # (2, 8, 1000)
    print(f"总参数量:     {sum(p.numel() for p in model.parameters()):,}")
    print()

    # 模拟预测：取 logits 最大值对应的词
    predictions = logits.argmax(dim=-1)
    print(f"预测 token ID: {predictions[0].tolist()}")
