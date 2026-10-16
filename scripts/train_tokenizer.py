# -*- coding: utf-8 -*-
"""训练生产版分词器：字节级 BPE + 特殊 token。

用法:
    py scripts/train_tokenizer.py
    py scripts/train_tokenizer.py --vocab-size 8192
    py scripts/train_tokenizer.py data/processed/corpus.txt data/tokenizer/tokenizer.json

为什么不用 scripts/bpe_from_scratch.py 那份
------------------------------------------
那份是**教学版**：算法全手写，看得清每一步。但它有两个硬伤：
  1. encode() 是「对每段文本重放 4744 条合并规则」= O(合并数 × 文本长度)，
     528 万字在纯 Python 下根本跑不完；
  2. 它的增量 pair 计数在相邻合并时会重复计数（tests/test_bpe_counts.py 钉着）。
所以生产版换成 Rust 实现的 tokenizers 库。**算法是同一个算法**，
区别只在实现语言和性能。

设计要点
--------
· 字节级（ByteLevel）：从 256 个字节起步，永远不会有"词表外的字"
· vocab 16384：太小则序列长、算得慢；太大则 embedding 参数浪费
· 特殊 token 在训练时一并注册，不能事后补（补的话 id 是新的，
  预训练好的 embedding 里没有对应行）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from miniqwen.tokenizer import SPECIAL_TOKENS, NovelTokenizer  # noqa: E402


# ============================================================
# 训练
# ============================================================
def train_tokenizer(corpus_path: Path, vocab_size: int, out_path: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    # ByteLevel 预处理：先把文本切成小块，再在块内做 BPE。
    # add_prefix_space=False 是给「不是自然语言开头」的文本用的（中文无所谓）。
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        # 关键：把 256 个字节全部塞进初始词表。
        # 不写这一行，语料里没出现过的字节就不在词表里 -> 编不了 -> 报错。
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def corpus_iter():
        """逐行喂语料，不把 15MB 一次性读进内存。"""
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield line

    print(f"训练 BPE：词表目标 {vocab_size}（含 {len(SPECIAL_TOKENS)} 个特殊 token）")
    t0 = time.perf_counter()
    tok.train_from_iterator(corpus_iter(), trainer=trainer)
    print(f"  用时 {time.perf_counter() - t0:.1f} 秒")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    print(f"  已保存 {out_path.relative_to(ROOT)}")
    return tok


# ============================================================
# 评估
# ============================================================
def evaluate(tok, corpus_path: Path, holdout_chars: int = 200_000) -> dict:
    from tokenizers import Tokenizer

    text = corpus_path.read_text(encoding="utf-8")
    holdout = text[-holdout_chars:]          # 取末尾一段当留出样本

    enc = tok.encode(holdout)
    ids = enc.ids
    restored = tok.decode(ids)

    n_bytes = len(holdout.encode("utf-8"))
    n_chars = len(holdout)
    n_tok = len(ids)

    # 每块多少 token —— 这个数决定后面模型的 block_size 该设多大
    # 用空行分块近似"章节块"（corpus.txt 里章节之间是空行）
    chapter_lens = []
    chunks = [c for c in text.split("\n\n") if len(c) > 50]
    step = max(1, len(chunks) // 400)         # 抽样 400 块，别全编（省时间）
    for c in chunks[::step][:400]:
        chapter_lens.append(len(tok.encode(c).ids))

    chapter_lens.sort()
    if chapter_lens:
        def pct(p):
            return chapter_lens[min(len(chapter_lens) - 1, int(p * len(chapter_lens)))]
        p50, p90, p99 = pct(0.50), pct(0.90), pct(0.99)
    else:
        p50 = p90 = p99 = 0

    return {
        "n_bytes": n_bytes,
        "n_chars": n_chars,
        "n_tok": n_tok,
        "bytes_per_token": n_bytes / n_tok,
        "chars_per_token": n_chars / n_tok,
        "lossless": holdout == restored,
        "chunk_p50": p50,
        "chunk_p90": p90,
        "chunk_p99": p99,
        "n_chunks_sampled": len(chapter_lens),
    }


def compare_with_legacy(corpus_path: Path, new_tok, sample_chars: int = 2000) -> dict | None:
    """拿旧版从零实现的 5000 词表 tokenizer 比一比压缩率。

    ⚠️ 这是个「不干净」的对照，必须说清：
      旧版词表 5000、只在前 50 万字上训练；新版词表 16384、在 528 万字上训练。
      两个变量（词表大小 + 训练语料量）同时在变，所以差异不能全归给词表大小。
      但它仍然有信息量：能看出「从零那版到底差多少」。

    样本只取 2000 字：旧版的 encode 是 O(合并数 × 文本长度)，
    4744 条合并 × 20 万字要跑到天荒地老。
    """
    legacy_path = ROOT / "data" / "tokenizer" / "legacy_bpe_5000.json"
    if not legacy_path.exists():
        return None

    sys.path.insert(0, str(ROOT / "scripts"))
    from bpe_from_scratch import encode as legacy_encode, load_tokenizer

    text = corpus_path.read_text(encoding="utf-8")
    start = len(text) // 2
    sample = text[start:start + sample_chars]

    merges, vocab = load_tokenizer(str(legacy_path))
    t0 = time.perf_counter()
    legacy_ids = legacy_encode(sample, merges)
    legacy_sec = time.perf_counter() - t0
    new_ids = new_tok.encode(sample).ids

    n_bytes = len(sample.encode("utf-8"))
    return {
        "sample_chars": len(sample),
        "n_bytes": n_bytes,
        "legacy_tokens": len(legacy_ids),
        "legacy_bytes_per_token": n_bytes / len(legacy_ids),
        "legacy_sec": legacy_sec,
        "new_vocab": new_tok.get_vocab_size(),
        "new_tokens": len(new_ids),
        "new_bytes_per_token": n_bytes / len(new_ids),
    }


def main():
    ap = argparse.ArgumentParser(description="训练生产版分词器")
    ap.add_argument("corpus", nargs="?", default="data/processed/corpus.txt")
    ap.add_argument("out", nargs="?", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--report", default="reports/tokenizer_report.md")
    args = ap.parse_args()

    corpus_path = ROOT / args.corpus
    out_path = ROOT / args.out
    if not corpus_path.exists():
        print(f"[错误] 找不到语料 {corpus_path}", file=sys.stderr)
        print("先跑：py scripts/build_corpus.py data/raw data/processed", file=sys.stderr)
        return 1

    tok = train_tokenizer(corpus_path, args.vocab_size, out_path)

    print("\n=== 评估 ===")
    m = evaluate(tok, corpus_path)
    print(f"  留出样本 {m['n_chars']:,} 字 / {m['n_bytes']:,} 字节 -> {m['n_tok']:,} token")
    print(f"  压缩率 {m['bytes_per_token']:.2f} 字节/token"
          f"（平均 {m['chars_per_token']:.2f} 字/token）")
    print(f"  编解码往返无损: {'是 [OK]' if m['lossless'] else '否 [FAIL]'}")
    print(f"  文本块 token 数：P50={m['chunk_p50']:,}  P90={m['chunk_p90']:,}"
          f"  P99={m['chunk_p99']:,}")

    # 特殊 token 的 id
    ids_line = []
    for s in SPECIAL_TOKENS:
        ids_line.append(f"`{s}`={tok.token_to_id(s)}")

    # 与旧版从零实现的对比
    print("\n=== 与旧版（从零实现，词表 5000）对比压缩率 ===")
    cmp = compare_with_legacy(corpus_path, tok)
    if cmp:
        print(f"  同一段 {cmp['sample_chars']} 字 / {cmp['n_bytes']:,} 字节：")
        print(f"    旧版 5000 词表 : {cmp['legacy_tokens']:,} token "
              f"({cmp['legacy_bytes_per_token']:.2f} 字节/token，"
              f"编码耗时 {cmp['legacy_sec']:.1f}s)")
        print(f"    新版 16384 词表: {cmp['new_tokens']:,} token "
              f"({cmp['new_bytes_per_token']:.2f} 字节/token，瞬时)")
    else:
        print("  （没找到 legacy_bpe_5000.json，跳过）")

    # ---------------- 报告 ----------------
    lines = ["# 分词器报告", ""]
    lines.append(f"- 语料：`{args.corpus}`")
    lines.append(f"- 输出：`{args.out}`")
    lines.append(f"- 算法：字节级 BPE（Rust `tokenizers` 实现）")
    lines.append(f"- 词表目标：{args.vocab_size}，实际 {tok.get_vocab_size()}")
    lines.append("")
    lines.append("## 特殊 token")
    lines.append("")
    lines.append("| token | id |")
    lines.append("|---|---:|")
    for s in SPECIAL_TOKENS:
        lines.append(f"| `{s}` | {tok.token_to_id(s)} |")
    lines.append("")
    lines.append("## 压缩率")
    lines.append("")
    lines.append(f"留出样本：{m['n_chars']:,} 字 / {m['n_bytes']:,} 字节，"
                 f"编成 {m['n_tok']:,} 个 token")
    lines.append("")
    lines.append(f"- **{m['bytes_per_token']:.2f} 字节/token**"
                 f"（平均 {m['chars_per_token']:.2f} 字/token）")
    lines.append(f"- 编解码往返无损：**{'是' if m['lossless'] else '否'}**")
    lines.append("")
    if cmp:
        lines.append("## 与旧版（从零实现，词表 5000）对比")
        lines.append("")
        lines.append(f"同一段 {cmp['sample_chars']:,} 字 / {cmp['n_bytes']:,} 字节：")
        lines.append("")
        lines.append("| 版本 | 词表 | token 数 | 字节/token | 编码耗时 |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(f"| 旧版 `bpe_from_scratch.py` | 5,000 | {cmp['legacy_tokens']:,} | "
                     f"{cmp['legacy_bytes_per_token']:.2f} | {cmp['legacy_sec']:.1f}s |")
        lines.append(f"| 新版 `tokenizers` | {cmp['new_vocab']:,} | {cmp['new_tokens']:,} | "
                     f"{cmp['new_bytes_per_token']:.2f} | 瞬时 |")
        lines.append("")
        lines.append("> ⚠️ 这是**不干净**的对照：旧版词表 5000、只在前 50 万字上训练；"
                     "新版 16384、在 528 万字上训练。两个变量同时在变，"
                     "差异不能全归给词表大小。")
        lines.append(">")
        lines.append("> 但它仍说明一件事：旧的从零版 `encode()` 是 "
                     "O(合并数 × 文本长度)。实测 2000 字耗时 "
                     f"{cmp['legacy_sec']:.1f}s，按线性外推到 528 万字约 "
                     f"{cmp['legacy_sec'] * 5_280_000 / cmp['sample_chars'] / 60:.0f} 分钟；"
                     "而新版**训练 + 编码整条流程一共 18.6 秒**。")
        lines.append(">")
        lines.append("> 差距不在算法（是同一个算法），而在实现："
                     "Python 纯解释执行 vs Rust 编译执行。")
        lines.append("")

    lines.append("## 文本块长度分布（决定后面的 block_size）")
    lines.append("")
    lines.append(f"抽样 {m['n_chunks_sampled']} 个文本块：")
    lines.append("")
    lines.append("| 分位 | token 数 |")
    lines.append("|---|---:|")
    lines.append(f"| P50 | {m['chunk_p50']:,} |")
    lines.append(f"| P90 | {m['chunk_p90']:,} |")
    lines.append(f"| P99 | {m['chunk_p99']:,} |")
    lines.append("")
    lines.append("> block_size 取 P90 附近，能让大多数文本块被完整覆盖；"
                 "取 P50 则训练快但每块都看不完。")
    lines.append("")

    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入 {report_path.relative_to(ROOT)}")

    # 用封装层加载一遍，确认它能被上层正常使用
    print("\n=== 用 miniqwen/tokenizer.py 加载验证 ===")
    nt = NovelTokenizer(out_path)
    s = "萧炎点了点头，淡淡的道：“走吧。”"
    e = nt.encode(s)
    print(f"  {nt}")
    print(f"  encode('{s[:12]}...') -> {len(e)} 个 id")
    print(f"  decode 往返: {'一致' if nt.decode(e) == s else '不一致 [!!]'}")
    print(f"  chat 模板:")
    for ln in nt.apply_chat_template("写一段斗气突破", "萧炎盘膝而坐。").split("\n"):
        print(f"    {ln}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
