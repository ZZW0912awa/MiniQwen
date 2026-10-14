# -*- coding: utf-8 -*-
"""L1.5：Trigram 语言模型 —— 亲手摸到 n-gram 的天花板

从 L1 Bigram 改过来的实验版：预测下一个字符时用「前 2 个」字符。

做法：把 (前一字符, 当前字符) 拼成组合下标  prev * V + cur，
查找表从 (65, 65) 变成 (65*65, 65) = (4225, 65)。

[2026-10-14] 实验记录：
  - Bigram 基线  ：参数量   4,225   train 2.4429 / val 2.5086
  - Trigram 结果 ：参数量 274,625   train 2.0773 / val 2.2705
  - 参数翻 65 倍，val 只降 0.24 —— 收益在快速递减。

  那继续加长上下文行不行？用「计数法 n-gram」把 N 从 0 扫到 6
  （见 l1_ngram_ceiling.py，不用神经网络，直接数条件概率）：
      上下文长度  0     1     2     3     4     5     6
      val 交叉熵 3.35  2.48  2.05  1.79  1.78  2.00  2.30
  最好的是长度 3~4；到 5、6 【反而变差】。根因是结构性的：
    · 表大小按 V^(N+1) 指数爆炸 —— N=6 要 4.9 万亿个参数，
      而实测只出现过 26 万个上下文，99.99999% 的行是空的；
    · val 里 20.8% 的上下文训练时从没出现过，只能瞎猜。
  结论：n-gram 是「枚举 + 记忆」，加长上下文只会撞上稀疏墙。
  L2 注意力换的是另一条路：不枚举组合，而是学会「看哪些位置、看多重」，
  用固定大小的表示压缩任意长度的上下文。

  另注：神经 Trigram（2.2705）反而比计数法 Trigram（2.0471）差，
  不是模型不行，是**训练量不够** —— 10000 步 × 32 token = 32 万次参数更新，
  而计数法一次就看完 100 万 token。

运行:
    py ladder/l1_trigram.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接复用 L0 的数据管线。
import l0_data as L0

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# 模型
# ============================================================
class TrigramLanguageModel(nn.Module):
    """Bigram 的扩展版：用前 2 个字符预测下一个。

    唯一的参数变成一张 (V*V, V) 的表：
        第 (prev*V + cur) 行 = 「前两个字符是 (prev, cur) 时，下一个字符的评分」
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # (V*V, V) 的表：行下标是「字符对」的组合编号
        self.pair_table = nn.Embedding(vocab_size * vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """idx: (B, T) 整数张量。返回 logits (B, T, C) 和 loss。"""
        B, T = idx.shape
        V = self.vocab_size

        # [2026-10-14] 构造组合下标：每个位置用 (前一字符, 当前字符) 两个 id。
        #   左边补一列 0 再截掉尾巴 → prev[t] = idx[t-1]（整体右移一位）
        #   第 0 个位置没有"前一个"，用 0 占位（反正是学习出来的表，无碍）。
        prev = torch.cat(
            [torch.zeros(B, 1, dtype=torch.long, device=idx.device), idx[:, :-1]],
            dim=1,
        )                                        # (B, T)
        pair_idx = prev * V + idx                # 组合编号：(前, 现) → 0..V²-1

        # 查表：(B, T) → (B, T, C)
        logits = self.pair_table(pair_idx)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """自回归采样：一次生成一个字符，接到末尾，再喂回自己。

        注意：这个函数一行没改——逻辑都收在 forward 里，
        换成 trigram 后 generate 自动跟着升级。这就是封装的好处。
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                # (B, T, C)
            logits = logits[:, -1, :]            # 只留最后位置 -> (B, C)
            probs = F.softmax(logits, dim=-1)    # 沿类别维归一化
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率抽样
            idx = torch.cat([idx, idx_next], dim=1)             # 接到末尾
        return idx


# ============================================================
# 训练循环
# ============================================================
def estimate_loss(model, eval_iters=50):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = L0.get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(42)

    model = TrigramLanguageModel(L0.vocab_size).to(L0.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}（就是 {L0.vocab_size}² × {L0.vocab_size} 那张表）")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    print("\n=== 训练 ===")
    for step in range(10001):
        xb, yb = L0.get_batch("train")
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            m = estimate_loss(model)
            print(f"  step {step:5d}  train {m['train']:.4f}   val {m['val']:.4f}")

    print("\n=== 最终结果 ===")
    m = estimate_loss(model)
    print(f"  train loss = {m['train']:.4f}")
    print(f"  val   loss = {m['val']:.4f}")
    print(f"  随机基线   = 4.174")
    print(f"  Bigram基线 = 2.4429 / 2.5086   （trigram 应该比它低，但低不了太多）")

    print("\n=== 生成 300 个字符 ===")
    context = torch.zeros((1, 1), dtype=torch.long, device=L0.device)
    out_ids = model.generate(context, max_new_tokens=300)[0].tolist()
    print(L0.decode(out_ids))

    print("\n" + "=" * 60)
    print("请把这两个数字填进 ladder/README.md 的 loss 表：")
    print(f"  L1.5 Trigram | 参数量 {n_params:,} | train {m['train']:.4f} | val {m['val']:.4f}")


if __name__ == "__main__":
    main()
