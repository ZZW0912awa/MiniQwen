# -*- coding: utf-8 -*-
"""L0：数据 + 字符级 tokenizer + get_batch

【你要写的只有 get_batch() 一个函数】，其余是框架。

运行:
    py ladder/l0_data.py

写完的验收标准（脚本最后会自动自检）：
    1. x 和 y 的形状都是 (batch_size, block_size)
    2. **y 是 x 右移一位** —— 这一条 checkpoint 通过了，才算真懂自回归训练的数据形态
"""

import math
import sys
from pathlib import Path

import torch

# Windows 控制台默认 cp936，强制 UTF-8 输出，免得中文提示变乱码
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA = Path(__file__).resolve().parent / "data" / "input.txt"

# ============================================================
# 读数据
# ============================================================
text = DATA.read_text(encoding="utf-8")
print(f"语料长度: {len(text):,} 字符")

# ============================================================
# 字符级 tokenizer
# ============================================================
# ⚠️ 教学够用，但正式项目里有两个坑，先埋个印象：
#   1. sorted(set(text)) 的 id 分配**依赖语料**——换一份语料词表就变了，
#      之前训好的权重全部作废。正式版必须把词表存盘（我们后面用 tokenizer.json）。
#   2. encode 遇到词表里没有的字符会直接 KeyError。正式版要有 <unk> 兜底。
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


print(f"词表大小: {vocab_size}")
print(f"roundtrip 检查: {encode('hello')} -> {decode(encode('hello'))!r}")

# 随机乱猜的理论 loss = ln(词表大小)，这是必须打败的下界
print(f"随机基线 loss = ln({vocab_size}) = {math.log(vocab_size):.3f}")
print()

# ============================================================
# 转 tensor + 切分训练/验证集
# ============================================================
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"训练集 {len(train_data):,} token / 验证集 {len(val_data):,} token")

block_size = 8     # 上下文长度：用前 8 个字符预测第 9 个
batch_size = 4     # 一次取 4 段
torch.manual_seed(42)


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """从训练集/验证集里随机取一个 batch。

    要返回:
        x: (batch_size, block_size) 的 long tensor
        y: (batch_size, block_size) 的 long tensor，且 y 是 x 右移一位

    为什么 y 是 x 右移一位：语言模型的训练目标是"看着前 t 个 token，预测第 t+1 个"。
    所以同一段文本，输入是 [t0..t7]，目标就是 [t1..t8]——**错开一位**。
    这样一来，一段 8 个字符的文本就贡献了 8 个训练样本（每个位置都是一个预测任务），
    数据利用率极高，这也是自回归训练便宜的原因。

    提示:
        starts = torch.randint(0, len(d) - block_size, (batch_size,))
        x = torch.stack([d[s : s + block_size] for s in starts])
        y = torch.stack([d[s + 1 : s + 1 + block_size] for s in starts])
        注意起点范围要留够 block_size + 1 的长度，别越界。

    TODO: 你来写
    """
    raise NotImplementedError("把 get_batch 写完")


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    xb, yb = get_batch("train")

    assert xb.shape == (batch_size, block_size), f"x 形状不对: {tuple(xb.shape)}"
    assert yb.shape == (batch_size, block_size), f"y 形状不对: {tuple(yb.shape)}"

    print(f"\nx shape: {tuple(xb.shape)}   y shape: {tuple(yb.shape)}")
    for i in range(min(2, batch_size)):
        print(f"--- 第 {i} 段 ---")
        print("x:", repr(decode(xb[i].tolist())))
        print("y:", repr(decode(yb[i].tolist())))

    # 核心自检：把 y 的最后一列去掉，必须等于 x 去掉第一列
    assert torch.equal(yb[:, :-1], xb[:, 1:]), (
        "y 不是 x 右移一位，回去检查切片下标"
    )
    print("\n[OK] 自检通过：y 确实是 x 右移一位")
