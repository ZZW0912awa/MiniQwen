# -*- coding: utf-8 -*-
"""L1：Bigram 语言模型 —— 整个阶梯上最低的一级

目标：先跑通「模型 → 损失 → 反向传播 → 采样」这条完整链路，
      把 loss 从 4.174 压到 ~2.5，并生成出「像英文的乱码」。

它没有注意力、没有隐藏层，**全部参数只是一张 (65, 65) 的查找表**：
    第 i 行 = 「当前字符是 i 时，下一个字符该给多少分」
之所以叫 bigram，是因为预测下一个字符时只用了「紧邻的前 1 个」字符。

本文件只留了两个 TODO：forward 和 generate。
训练循环、评测、采样演示都写好了——那一层不是 L1 要学的东西。

运行:
    py ladder/l1_bigram.py

验收标准（三条都满足才算过）:
    1. 训练 loss 从 4.174 明显下降，落到 2.5 附近
    2. val loss 同步下降（说明模型学到了通用规律，不是死记硬背训练数据）
    3. 生成的文本「像英文但没意义」—— 这就对了，Bigram 只能做到这一步
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接复用 L0 的数据管线。
# 能这么写是因为：跑 `py ladder/l1_bigram.py` 时，Python 会自动把
# 「脚本所在目录」放进 sys.path —— 所以能 import 到同目录的 l0_data。
# （还记得 conftest.py 那一课吗？那里是我们手动往 sys.path 里加的。）
import l0_data as L0

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# 模型
# ============================================================
class BigramLanguageModel(nn.Module):
    """最小可训练的语言模型。

    继承 nn.Module 之后，PyTorch 会自动帮你：
      - 收集所有子模块和参数（model.parameters() 才拿得到东西）
      - 支持 model.to(device) / model.train() / model.eval()
      - 记录计算图，loss.backward() 才能自动求梯度
    """

    def __init__(self, vocab_size):
        super().__init__()          # 必须调父类构造，否则参数注册不生效
        # 唯一的参数：一张 (vocab_size, vocab_size) 的表
        # nn.Embedding(V, V) 的第 i 行就是「下一个字符的评分向量」
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # --------------------------------------------------------
    # TODO 1：前向
    # --------------------------------------------------------
    def forward(self, idx, targets=None):
        """算出每个位置对「下一个字符」的评分，以及损失。

        参数
            idx     : (B, T) 整数张量 —— B 个样本，每个 T 个字符
            targets : (B, T) 整数张量，或 None（只推理、不算损失时）

        返回
            logits  : (B, T, vocab_size)
            loss    : 标量张量；targets 为 None 时是 None

        提示
            1. 查表：logits = self.token_embedding_table(idx)
               因为表本身就是 (V, V)，查出来的直接就是「下一个字符的评分」。
            2. **cross_entropy 只吃二维输入**（样本数, 类别数）。
               现在 logits 是三维 (B, T, C)，targets 是二维 (B, T)。
               要先拍平：
                   logits  -> (B*T, C)     用 .view(-1, C)
                   targets -> (B*T,)       用 .view(-1)
               （-1 的意思是「这一维你自己算出来」。B*T 个位置，
                 每个位置都是一次独立的「预测下一个字符」，所以可以拍平。）
            3. loss = F.cross_entropy(拍平后的 logits, 拍平后的 targets)
            4. targets 是 None 时不要算 loss，直接 return logits, None
        """
        raise NotImplementedError("TODO 1：把 forward 写完")

    # --------------------------------------------------------
    # TODO 2：采样生成
    # --------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """自回归采样：一次生成一个字符，接到末尾，再喂回自己。

        这个 @torch.no_grad() 是说「这段不需要算梯度」——
        推理时没有反向传播，关掉梯度能省一半显存、快不少。

        idx : (B, T) 起始上下文
        返回: (B, T + max_new_tokens)

        循环 max_new_tokens 次，每一步：
            1. logits, _ = self(idx)                前向
            2. logits = logits[:, -1, :]            只留最后一个位置 -> (B, C)
               为什么只要最后一个：上下文里更早的位置预测的是它们各自的
               「下一个」，我们只关心当前位置的下一个字符。
            3. probs = F.softmax(logits, dim=-1)    把评分变成概率
            4. idx_next = torch.multinomial(probs, num_samples=1)
               按概率**随机抽一个**（不是取最大的那个！取最大就是贪心，容易复读）
            5. idx = torch.cat([idx, idx_next], dim=1)   接到序列末尾，进入下一轮

        注意两个 dim：softmax 要沿「类别」那一维（dim=-1），
        cat 要沿「时间」那一维（dim=1）。写错**不会报错**，但结果全乱。
        """
        raise NotImplementedError("TODO 2：把 generate 写完")


# ============================================================
# 训练循环（已给全，L1 不考这个）
# ============================================================
def estimate_loss(model, eval_iters=50):
    """在 train / val 上各算若干批的平均 loss。

    注意 model.eval() 和 model.train() 这一对：
    它们切换「训练模式 / 评估模式」。现在用不到差别，
    但以后有了 Dropout，评估时必须关掉它，否则每次评估结果都在抖。
    """
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

    model = BigramLanguageModel(L0.vocab_size).to(L0.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}（就是 {L0.vocab_size} × {L0.vocab_size} 那张表）")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    print("\n=== 训练 ===")
    for step in range(10001):
        xb, yb = L0.get_batch("train")
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)   # 清空上一步的梯度
        loss.backward()                         # 反向传播，算梯度
        optimizer.step()                        # 按梯度更新参数

        if step % 2000 == 0:
            m = estimate_loss(model)
            print(f"  step {step:5d}  train {m['train']:.4f}   val {m['val']:.4f}")

    print("\n=== 最终结果 ===")
    m = estimate_loss(model)
    print(f"  train loss = {m['train']:.4f}")
    print(f"  val   loss = {m['val']:.4f}")
    print(f"  随机基线   = 4.174   （ln 65，必须打败它）")

    print("\n=== 生成 300 个字符 ===")
    # 用一个全 0 的上下文起步（token id 0 是本语料里最小的字符，通常是换行）
    context = torch.zeros((1, 1), dtype=torch.long, device=L0.device)
    out_ids = model.generate(context, max_new_tokens=300)[0].tolist()
    print(L0.decode(out_ids))

    print("\n" + "=" * 60)
    print("请把这两个数字填进 ladder/README.md 的 loss 表：")
    print(f"  L1 Bigram | 参数量 {n_params:,} | train {m['train']:.4f} | val {m['val']:.4f}")


if __name__ == "__main__":
    main()
