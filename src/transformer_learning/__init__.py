"""
transformer_learning - 从零开始实现 Transformer

模块结构：
    config.py               → 超参数配置
    attention.py             → 缩放点积注意力
    multihead_attention.py   → 多头注意力
    FFN.py                   → 前馈神经网络
    embedding.py             → 词嵌入 + 位置编码
    Encoder.py               → 编码器
    decoder.py               → 解码器
    transformer.py           → 完整 Transformer 模型
"""

from transformer_learning.config import ModelArgs
from transformer_learning.transformer import Transformer


def main() -> None:
    """入口函数：运行一个简单的 Transformer 前向传播演示。"""
    import torch

    print("🚀 Transformer Learning - 从零实现 Transformer")
    print("=" * 55)

    # 配置
    args = ModelArgs()
    args.vocab_size = 1000

    # 创建模型
    model = Transformer(args)

    # 模拟输入
    src = torch.randint(1, 1000, (1, 20))
    tgt = torch.randint(1, 1000, (1, 15))

    # 前向传播
    logits = model(src, tgt)

    print(f"模型配置:")
    print(f"  - 嵌入维度:   {args.dim}")
    print(f"  - 注意力头数: {args.n_heads}")
    print(f"  - 层数:       {args.n_layers}")
    print(f"  - FFN 中间维: {args.d_ff}")
    print(f"  - 词表大小:   {args.vocab_size}")
    print()
    print(f"输入: src={src.shape}, tgt={tgt.shape}")
    print(f"输出: logits={logits.shape}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("✅ 前向传播成功！")
