# -*- coding: utf-8 -*-
"""
mini_bpe.py v2 —— 从零实现 BPE Tokenizer（增量统计优化版）

用法:
    py bpe_tokenizer.py 语料.txt        # 前50万字训练，词表2万，随机段落评测

相比教学版的升级：
  1. 增量统计 —— 每轮合并后只更新「合并点附近」的 pair 计数，
     不再全量重扫语料。50万字 + 2万词表从几小时降到几分钟。
  2. 留出法评测 —— 训练只用前 50 万字，从后面的文本随机抽段做
     编码→解码往返测试，检验泛化能力。
"""

import sys
import json
import random
from difflib import SequenceMatcher
# 文件顶部 import 区加一行
from collections import defaultdict

# ============================================================
# 读语料（自动识别编码）
# ============================================================
def read_corpus(path):
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            print(f"（编码识别为 {enc}，全文 {len(text)} 字符）")
            return text
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码")


# ============================================================
# 文本 → UTF-8 字节序列
# ============================================================
def text_to_bytes(text):
    return list(text.encode("utf-8"))


# ============================================================
# 初始全量统计（只调用一次）
# ============================================================
def get_pair_counts(ids):
    counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# ============================================================
# 合并 + 增量更新计数（核心优化）
# ============================================================
def merge_pair_incremental(ids, pair, new_id, counts):
    """
    把所有 (a,b) 合并为 new_id，同时只更新受影响位置的计数：
      - 旧序列中被破坏的相邻对：计数减一
      - 新序列中合并点产生的新相邻对：计数加一
    其余位置的 pair 完全没动，不用管。
    """
    a, b = pair
    n = len(ids)
    new_ids = []
    positions = []          # 记录每个新 token 在新序列中的位置

    i = 0
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            # 被破坏的旧对（按旧序列的下标看邻居）
            if i > 0:
                counts[(ids[i - 1], a)] -= 1
            if i + 2 < n:
                counts[(b, ids[i + 2])] -= 1
            positions.append(len(new_ids))
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    # 新产生的新对（在新序列里看合并点的左右邻居）
    for j in positions:
        if j > 0:
            counts[(new_ids[j - 1], new_id)] += 1
        if j + 1 < len(new_ids):
            counts[(new_id, new_ids[j + 1])] += 1

    counts[pair] = 0        # (a,b) 已全部被合并
    return new_ids


# ============================================================
# 训练 BPE
# ============================================================
def train_bpe(text, vocab_size=20000):
    ids = text_to_bytes(text)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    counts = defaultdict(int, get_pair_counts(ids))   # 不存在的键自动按 0 处理
    initial_len = len(ids)

    num_merges = vocab_size - 256
    for i in range(num_merges):
        # 只在计数 > 0 的 pair 里找最高频
        best_pair, best_cnt = None, 0
        for p, c in counts.items():
            if c > best_cnt or (c == best_cnt and best_pair is not None
                                and c > 0 and p < best_pair):
                best_pair, best_cnt = p, c
        if best_pair is None or best_cnt < 2:
            print("没有可合并的 pair 了，提前结束")
            break

        new_id = 256 + i
        ids = merge_pair_incremental(ids, best_pair, new_id, counts)
        merges.append((best_pair, new_id))
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        if i < 15 or i % 300 == 0:
            token_bytes = vocab[new_id]
            try:
                show = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                show = repr(token_bytes)
            print(f"  合并 {i:5d}: 频率={best_cnt:6d}  内容={show}  "
                  f"序列长度={len(ids)}")

    print(f"\n训练完成: {len(merges)} 次合并")
    print(f"压缩效果: {initial_len} 字节 -> {len(ids)} token "
          f"(压缩率 {initial_len / len(ids):.2f}x)")
    return merges, vocab


# ============================================================
# 编码 / 解码
# ============================================================
def encode(text, merges):
    ids = text_to_bytes(text)
    for pair, new_id in merges:
        ids = merge_pair(ids, pair, new_id)
    return ids


def merge_pair(ids, pair, new_id):
    """编码用的朴素合并（不涉及计数，逻辑简单）"""
    a, b = pair
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def decode(ids, vocab):
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


# ============================================================
# 保存 / 加载
# ============================================================
def save_tokenizer(merges, vocab, path="tokenizer.json"):
    data = {
        "merges": [[a, b, new_id] for (a, b), new_id in merges],
        "vocab": {str(k): list(v) for k, v in vocab.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"已保存到 {path}")


def load_tokenizer(path="tokenizer.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merges = [((a, b), new_id) for a, b, new_id in data["merges"]]
    vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
    return merges, vocab


# ============================================================
# 评测：随机抽取未见过的段落，编码→解码往返
# ============================================================
def evaluate(full_text, train_chars, merges, vocab, sample_len=300):
    """从训练集之外随机抽一段，测试往返准确率"""
    rest = full_text[train_chars:]
    if len(rest) < sample_len:
        rest = full_text           # 语料不够长就从全文抽
        print("（语料不足，从全文抽样）")

    start = random.randint(0, len(rest) - sample_len)
    sample = rest[start:start + sample_len]

    ids = encode(sample, merges)
    restored = decode(ids, vocab)

    # 逐字符一致率（BPE 是无损压缩，理论上应该 100%）
    char_acc = SequenceMatcher(None, sample, restored).ratio()
    exact = sample == restored

    print("\n=== 随机段落评测 ===")
    print(f"抽样位置: 全文第 {train_chars + start} 字符起 {sample_len} 字")
    print(f"原文开头: {sample[:50]}...")
    print(f"解码开头: {restored[:50]}...")
    print(f"逐字符一致率: {char_acc * 100:.2f}%")
    print(f"完全一致: {'[OK]' if exact else '[FAIL]'}")
    print(f"压缩: {len(sample.encode('utf-8'))} 字节 -> {len(ids)} token "
          f"({len(sample)} 字 => 每字 {len(ids)/len(sample):.2f} token)")


# ============================================================
# 主流程
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("用法: py bpe_tokenizer.py 语料.txt")
        sys.exit(1)

    full_text = read_corpus(sys.argv[1])

    TRAIN_CHARS = 500000     # 前 50 万字用于训练
    VOCAB_SIZE = 5000       # 词表 2 万

    train_text = full_text[:TRAIN_CHARS]
    print(f"训练集: 前 {len(train_text)} 字符  词表目标: {VOCAB_SIZE}")

    print("\n=== 训练 BPE ===")
    merges, vocab = train_bpe(train_text, VOCAB_SIZE)
    save_tokenizer(merges, vocab)

    evaluate(full_text, TRAIN_CHARS, merges, vocab)


if __name__ == "__main__":
    main()
