# -*- coding: utf-8 -*-
"""L1.5 附：用「计数法 n-gram」量出上下文长度的天花板

问题：既然 trigram（看前 2 个字符）比 bigram 好，
      那把上下文从 2 个、3 个、4 个……一直加下去，是不是就越来越好？

这个脚本不用神经网络，直接「数数」：统计训练语料里每个上下文后面
跟过哪些字符，得到条件概率，然后在验证集上算交叉熵。
这就是 n-gram 路线的**理论极限** —— 神经网络的 bigram/trigram
正是在逼近这张"数出来的"分布。

输出三组数字：
  1. 表大小 = V^(N+1)，看它怎么爆
  2. val 交叉熵，看提升什么时候停止、什么时候反弹
  3. val 里"上下文从没在训练集出现过"的比例，看稀疏墙在哪

运行:
    py ladder/l1_ngram_ceiling.py

实测结果（tinyshakespeare，1,115,394 字符，词表 65）见 ladder/README.md。
"""

import math
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA = Path(__file__).resolve().parent / "data" / "input.txt"

# 平滑系数：给没见过的 (上下文, 下一个字符) 留一点概率，防止 log(0)
K = 0.1
MAX_N = 6


def main():
    text = DATA.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in text]
    V = len(chars)

    n_split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n_split], ids[n_split:]
    print(f"语料 {len(ids):,} 字符，词表 {V}，"
          f"训练 {len(train_ids):,} / 验证 {len(val_ids):,}")
    print()

    # 单字频率：上下文没见过时用它兜底（退化成"只看单字频率"）
    uni = Counter(train_ids)
    uni_total = sum(uni.values())

    print(f"{'上下文长度':>6} {'理论表参数量':>16} {'实测上下文数':>13} "
          f"{'val 交叉熵':>11} {'未见上下文':>11}")
    print("-" * 72)

    for N in range(0, MAX_N + 1):
        # 上下文用「滚动整数编码」表示：把最近 N 个字符当成 V 进制数
        # （用整数当 key 比元组快得多，N=6 时差别很明显）
        mod = V ** N if N > 0 else 1

        # --- 训练集：统计 (上下文) 和 (上下文, 下一个字符) 的次数 ---
        ctx_counts = Counter()
        pair_counts = Counter()
        ctx = 0
        for i, x in enumerate(train_ids):
            if i >= N:
                ctx_counts[ctx] += 1
                pair_counts[(ctx, x)] += 1
            ctx = (ctx * V + x) % mod if mod > 1 else 0

        # --- 验证集：算交叉熵 ---
        loss = 0.0
        n = 0
        unseen = 0
        ctx = 0
        for i, x in enumerate(val_ids):
            if i >= N:
                c = ctx_counts.get(ctx, 0)
                if c == 0:
                    # 这个上下文训练时从没见过 -> 只能退回单字频率瞎猜
                    unseen += 1
                    p = (uni.get(x, 0) + 1) / (uni_total + V)
                else:
                    p = (pair_counts.get((ctx, x), 0) + K) / (c + K * V)
                loss += -math.log(p)
                n += 1
            ctx = (ctx * V + x) % mod if mod > 1 else 0

        print(f"{N:>6} {V ** N * V:>16,} {len(ctx_counts):>13,} "
              f"{loss / n:>11.4f} {100.0 * unseen / n:>10.1f}%")

    print()
    print("=" * 72)
    print("怎么读这张表")
    print("=" * 72)
    print("  · 上下文 1 -> 2 -> 3：交叉熵一路下降，说明「多看一点确实有用」")
    print("  · 上下文 3~4 之后：提升几乎消失，再往上【反而变差】")
    print("  · 表大小按 V^(N+1) 指数爆炸：上下文 6 就要 4.9 万亿个参数")
    print("    （而实测只出现过 26 万个上下文 —— 99.99999% 的行是空的）")
    print("  · 「未见上下文」比例一路涨：20% 的验证上下文训练时压根没出现过")
    print()
    print("  根因：n-gram 是「枚举 + 记忆」，它把每个上下文当成独立个体，")
    print("        无法共享统计强度。上下文越长越容易没见过，见没见过之间没有过渡。")
    print("  出路：L2 的注意力不枚举组合，而是学着「看哪些位置、看多重」，")
    print("        用固定大小的表示压缩任意长度的上下文 ——")
    print("        参数量随上下文线性增长（甚至不变），而不是指数爆炸。")


if __name__ == "__main__":
    main()
