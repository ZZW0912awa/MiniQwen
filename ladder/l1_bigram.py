# -*- coding: utf-8 -*-
"""L1：Bigram 语言模型 —— 整个阶梯上最低的一级

目标：先跑通「模型 → 损失 → 反向传播 → 采样」这条完整链路，
      把 loss 从 4.174 压到 ~2.5，并生成出「像英文的乱码」。

它没有注意力、没有隐藏层，**全部参数只是一张 (65, 65) 的查找表**：
    第 i 行 = 「当前字符是 i 时，下一个字符该给多少分」
之所以叫 bigram，是因为预测下一个字符时只用了「紧邻的前 1 个」字符。

★ 关于语料：这一级用 tinyshakespeare（英文，只有 65 种字符）。
  不是因为它"重要"，而是因为它小到能一眼看穿、跑几十秒就出结果，
  而且和 Karpathy 的视频完全对得上。
  换成中文语料（4,754 种字符）会让这张表变成 4,754² = 2260 万参数，
  学到的东西性质完全一样，只是慢几十倍 —— 所以先在这里把架构搞懂，
  到 L6 再整体换回中文。
  （再推一步：BPE 16k 词表的 bigram 表是 16384² = 2.68 亿参数。
    撑不住 —— 这就是必须有 L2 注意力的原因：词表一大，直接查表就爆炸，
    必须靠"低维表示 + 注意力"共享参数。）

运行:
    py ladder/l1_bigram.py

验收标准（三条都满足才算过）:
    1. 训练 loss 从 4.174 明显下降，落到 2.5 附近
    2. val loss 同步下降（说明学到了通用规律，不是死记硬背训练数据）
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
        # 注意：这里 embedding 维度 == vocab_size，不是巧合 ——
        #       因为我们就是要为每个可能的「下一个字符」各存一个分数。
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # --------------------------------------------------------
    # 前向
    # --------------------------------------------------------
    def forward(self, idx, targets=None):
        """算出每个位置对「下一个字符」的评分，以及损失。

        idx     : (B, T) 整数张量 —— B 个样本，每个 T 个字符
        targets : (B, T) 整数张量，或 None（只推理、不算损失时）

        返回 logits (B, T, C) 和 loss（targets 为 None 时 loss 是 None）
        """
        # ① 查表。idx 是 (B, T)，每个元素都是 0..64 的下标；
        #    Embedding 会把每个下标换成表里对应的那一行（长度 65 的向量）。
        #    (B, T) --查表--> (B, T, C)，C = vocab_size = 65
        logits = self.token_embedding_table(idx)

        # ② 只推理不算损失时，直接返回
        if targets is None:
            return logits, None

        # ③ cross_entropy 只接受二维输入「(样本数, 类别数)」，
        #    所以要把三维的 logits 拍平。
        B, T, C = logits.shape
        logits = logits.view(B * T, C)      # (B, T, C) -> (B*T, C)
        targets = targets.view(B * T)       # (B, T)    -> (B*T,)

        #    为什么可以拍平？因为语言模型把「每个位置预测下一个字符」
        #    当成一次独立的多分类任务，一共 B*T 个任务，各自算一次交叉熵，
        #    再取平均。拍平只是把它们摞在一起一次性算完。
        #
        #    .view() 要求内存连续；万一某步拿到的是非连续张量会报错，
        #    那时改用 .reshape() 即可（它会自动复制成连续再变形）。
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    # --------------------------------------------------------
    # 采样生成
    # --------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """自回归采样：一次生成一个字符，接到末尾，再喂回自己。

        @torch.no_grad() 是说「这段不算梯度」——
        推理时没有反向传播，关掉梯度能省显存、也更快。

        idx : (B, T) 起始上下文
        返回: (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # ① 前向，拿到 (B, T, C)
            logits, _ = self(idx)

            # ② 只留最后一个位置 -> (B, C)
            #    上下文里更早的位置预测的是它们各自的「下一个」，
            #    我们现在只关心「整个上下文之后」的下一个字符。
            logits = logits[:, -1, :]

            # ③ 把评分变成概率分布（每行加起来等于 1）
            probs = F.softmax(logits, dim=-1)

            # ④ 按概率随机抽一个下标 -> (B, 1)
            #    注意不是 argmax！取最大值是"贪心"，模型会一直重复同一句话。
            #    按概率抽才能有多样性。
            idx_next = torch.multinomial(probs, num_samples=1)

            # ⑤ 接到序列末尾 -> (B, T+1)，进入下一轮
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    # ⚠️ 留意这里的效率问题：每生成一个字符，都要把整个上下文重新前向一遍
    #    （第 ① 步），所以总计算量是 O(T²)。上下文一长就慢得离谱。
    #    这正是阶段 5 要做的 **KV Cache** 解决的 —— 缓存已经算过的 K/V，
    #    新字符只算增量，能快十几倍。现在先记住这个坑。


# ============================================================
# 训练循环（不属于 L1 的学习目标，直接给全）
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
    # 用一个全 0 的上下文起步（token id 0 是本语料里最小的字符）
    context = torch.zeros((1, 1), dtype=torch.long, device=L0.device)
    out_ids = model.generate(context, max_new_tokens=300)[0].tolist()
    print(L0.decode(out_ids))

    print("\n" + "=" * 60)
    print("请把这两个数字填进 ladder/README.md 的 loss 表：")
    print(f"  L1 Bigram | 参数量 {n_params:,} | train {m['train']:.4f} | val {m['val']:.4f}")


if __name__ == "__main__":
    main()
