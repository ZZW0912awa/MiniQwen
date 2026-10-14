# -*- coding: utf-8 -*-
"""
bpe_from_scratch.py —— 从零实现字节级 BPE Tokenizer（增量统计优化版）

用法:
    py scripts/bpe_from_scratch.py 语料.txt [训练字数] [词表大小]
    py scripts/bpe_from_scratch.py data/processed/corpus.txt 100000 3000

相比第一版的升级：
  1. 增量统计 —— 每轮合并后只更新「合并点附近」的 pair 计数，
     不再全量重扫语料，把训练时间从小时级压到分钟级。
  2. 留出法评测 —— 训练只用前 N 字，从后面的文本随机抽段做
     编码→解码往返测试，看看对没见过的文本是否也有效。
  3. 进度与 ETA —— 每 100 轮报一次进度和预计剩余时间。

注意：默认参数是「小而快」（10 万字 / 词表 3000），方便几分钟跑完看结果；
     要训真正能用的词表就在命令行传更大的值，例如 500000 16384。
"""

import sys
import json
import time
import random
from difflib import SequenceMatcher
from collections import defaultdict


# ============================================================
# 读语料（自动识别编码）
# ============================================================
# [2026-10-14] 知识点：编码探测的顺序有讲究。这里把 utf-16 也放进试错列表，
#              而它没有 BOM 时【不会报错】，会按小端硬解出乱码 —— 是个隐患。
#              miniqwen/corpus.py 里已改成「BOM 优先，再 utf-8 → gbk」，
#              以后以那份为准（中文语料实测很少踩到这个坑，因为 UTF-8 那关
#              会把 GBK 文件拦住、反之亦然）。
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
# [2026-10-14] 知识点：BPE 从 256 个字节起步而不是汉字——任何文本都能编成
#              UTF-8 字节，永远不会遇到"词表外"的字，算法本身完全不用懂中文。
#              "萧" → b'\xe8\x90\xa7' → [232, 144, 167]（一汉字 = 3 字节）。
def text_to_bytes(text):
    return list(text.encode("utf-8"))


# ============================================================
# 初始全量统计（只调用一次）
# ============================================================
# [2026-10-14] 知识点：counts.get(pair, 0) + 1 是字典计数套路
#              （键不存在时 get 返回 0，不会 KeyError）。键是二元组 (左id, 右id)。
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

    [2026-10-14] 知识点：为什么折腾增量？朴素做法每轮全量重扫，代价是
    「语料长度 × 合并轮数」——50 万汉字 ≈ 150 万字节，配上万轮合并就是
    千亿级别的比较，纯 Python 根本跑不完。洞察：一轮合并只影响合并点
    附近的计数，其余位置纹丝不动。

    已知缺陷：同一轮里相邻位置同时发生合并时（典型是 "ababab" 这种交替
    模式），中间那个 pair 会被左右两次合并各扣一次账、各开一次账。
    实测 merge (1,2) 之后：(2,1) 的计数变成 -2（真实是 0），
    新对 (256,256) 算成 4（真实是 2）—— 相对误差能到 100%，不是"微小误差"。
    后果必须说清：这只影响【合并顺序】（压缩率略差一点），不影响正确性 ——
    encode 是重放规则，编码解码往返依然 100% 无损。
    仓库里用 tests/test_bpe_counts.py 的 xfail 测试把它钉住了。
    """
    a, b = pair             # 元组解包：a, b = (左, 右)
    n = len(ids)
    new_ids = []
    positions = []          # 记录每个新 token 在新序列中的位置

    i = 0
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            # 销账：被合并点破坏的左右旧对（按旧序列下标看邻居）
            if i > 0:
                counts[(ids[i - 1], a)] -= 1
            if i + 2 < n:
                counts[(b, ids[i + 2])] -= 1
            positions.append(len(new_ids))
            new_ids.append(new_id)
            i += 2          # 两格并一格，跳过被吃掉的第二个
        else:
            new_ids.append(ids[i])
            i += 1

    # 开账：合并点在新序列里产生的新相邻对
    for j in positions:
        if j > 0:
            counts[(new_ids[j - 1], new_id)] += 1
        if j + 1 < len(new_ids):
            counts[(new_id, new_ids[j + 1])] += 1

    counts[pair] = 0        # (a,b) 已全部被合并，清零防它再被选中
    return new_ids


# ============================================================
# 训练 BPE
# ============================================================
# [2026-10-14] 知识点：BPE 核心循环——每轮找最高频相邻对，合并成新 token。
#              token 的 id 就是合并顺序（256+i），不是人为规定的；
#              高频词（萧炎、斗气）会被频率统计自动拼出来。
def train_bpe(text, vocab_size=20000):
    ids = text_to_bytes(text)
    # [2026-10-14] 知识点：字典推导式 {键: 值 for ...}，一行建初始词表
    #              {0: b'\x00', ..., 255: b'\xff'}。bytes([i]) 是单字节。
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    # [2026-10-14] 知识点：defaultdict(int, 初始字典)——访问不存在的键自动给 0。
    #              我当初的 KeyError (155, 256) 就是没用它的缘故。
    counts = defaultdict(int, get_pair_counts(ids))   # 全量统计只这一次
    initial_len = len(ids)
    t0 = time.time()

    num_merges = vocab_size - 256
    for i in range(num_merges):
        # 打擂台找最高频对；平手时取字典序最小的 pair（先比左 id、再比右 id），
        # 让结果与字典遍历顺序无关，每次运行都一样，方便复现实验
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
        merges.append((best_pair, new_id))      # 训练的核心产出：哪对 → 几号
        # 新 token 的字节内容 = 左右两边递归拼接（bytes + bytes）
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        if i < 15 or i % 100 == 0:
            token_bytes = vocab[new_id]
            try:
                show = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # 合并到一半可能是半个汉字（字节没凑齐），解不了就显示字节
                show = repr(token_bytes)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (num_merges - i - 1)
            print(f"  合并 {i:5d}/{num_merges}: 频率={best_cnt:6d}  "
                  f"内容={show}  序列={len(ids)}  "
                  f"用时={elapsed:.0f}s  预计剩余={eta/60:.1f}分钟")

    print(f"\n训练完成: {len(merges)} 次合并")
    print(f"压缩效果: {initial_len} 字节 -> {len(ids)} token "
          f"(压缩率 {initial_len / len(ids):.2f}x)")
    return merges, vocab


# ============================================================
# 编码 / 解码
# ============================================================
# [2026-10-14] 知识点：编码 = 按训练时的顺序逐条【重放】合并规则。
#              顺序必须严格一致——后面的规则可能依赖前面合并出的 token。
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
    # 每个 id 查词表取字节 → b"".join 拼接 → UTF-8 解码回字符串。
    # errors="replace"：token 序列从中间截断时，半个汉字解不出就换成 �，
    # 不崩溃（生成到一半的场景会用到）。
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


# ============================================================
# 保存 / 加载
# ============================================================
# [2026-10-14] 知识点：JSON 只认 str/int/float/list/dict/bool/None，
#              元组、bytes、int 键都要先转换；load 时反向转回。
#              存的时候怎么转、读的时候就怎么转回来，不对称就是 bug。
def save_tokenizer(merges, vocab, path="tokenizer.json"):
    data = {
        "merges": [[a, b, new_id] for (a, b), new_id in merges],   # 元组→列表
        "vocab": {str(k): list(v) for k, v in vocab.items()},       # int键→str, bytes→list
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)   # 保留汉字，不转 \uXXXX
    print(f"已保存到 {path}")


def load_tokenizer(path="tokenizer.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merges = [((a, b), new_id) for a, b, new_id in data["merges"]]  # 列表→元组
    vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}    # str键→int, list→bytes
    return merges, vocab


# ============================================================
# 评测：随机抽取未见过的段落，编码→解码往返
# ============================================================
# [2026-10-14] 知识点：留出法——训练只用前 TRAIN_CHARS 个字符（默认 10 万），
#              评测从后面抽（模型没见过的数据才能测真实能力，和 train/test split 同思想）。
#              BPE 是无损压缩，往返必然 100%，所以准确率是预期不是惊喜；
#              真正有价值的指标是压缩率和 token/字（决定模型推理成本）。
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

    # [2026-10-14] 知识点：SequenceMatcher.ratio() 返回两个字符串的相似度 0~1
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
        print("用法: py scripts/bpe_from_scratch.py 语料.txt [训练字数] [词表大小]")
        print("示例: py scripts/bpe_from_scratch.py data/processed/corpus.txt 100000 3000")
        sys.exit(1)

    full_text = read_corpus(sys.argv[1])

    # [2026-10-14] 知识点：A if 条件 else B 给可选参数设默认值
    #              （正经做法是 argparse，这个脚本简单就用了 sys.argv）
    TRAIN_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    VOCAB_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

    train_text = full_text[:TRAIN_CHARS]
    print(f"训练集: 前 {len(train_text)} 字符  词表目标: {VOCAB_SIZE}")

    print("\n=== 训练 BPE ===")
    merges, vocab = train_bpe(train_text, VOCAB_SIZE)
    save_tokenizer(merges, vocab)

    evaluate(full_text, TRAIN_CHARS, merges, vocab)


# [2026-10-14] 知识点：__name__ == "__main__" 守门——直接运行才执行 main，
#              被 import 时不触发。
if __name__ == "__main__":
    main()
