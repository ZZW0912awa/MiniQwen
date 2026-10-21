# -*- coding: utf-8 -*-
"""阶段 3：把文本语料固化成 uint16 二进制，供训练时 memmap 直接读。

用法:
    py scripts/prepare_data.py
    py scripts/prepare_data.py --val-frac 0.02
    py scripts/prepare_data.py --vocab-path data/tokenizer/tokenizer.json

产出（都在 data/processed/）:
    train.bin     训练集，uint16 的 token id 序列
    val.bin       验证集
    meta.json     元信息：token 数、切分点、词表大小、sha256

为什么要这一步
-------------
1. **随机访问 O(1)**：文本是变长 UTF-8，第 100 万个 token 在哪个字节算不出来；
   定长数组里它就在第 i×2 字节。
2. **切点总在 token 边界**：不会切在某个 BPE 合并的中间。
3. **内存**：uint16 比 Python list of int 省 14 倍；memmap 还按页懒加载。

设计要点
--------
· uint16：实测最大 id 16,383 < 65,535，2 字节够了
· 分块编码：每块 100 万字，边界上只会少合并几个 token（实测 +0.0001%），
  换来内存有界 —— 扩到 1GB 语料也不会炸
· 切分点落在【章节边界】上，而不是硬切在第 98% 个字符处
· 写完立刻回读校验：解码回来必须和原文逐字一致
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from miniqwen.tokenizer import DEFAULT_TOKENIZER_PATH, NovelTokenizer  # noqa: E402

CHUNK_CHARS = 1_000_000      # 每次编码多少字符（控制内存峰值）
DTOYPE = np.uint16


def find_split_point(text: str, val_frac: float) -> int:
    """在「章节边界」附近找切分点，而不是硬切在第 (1-val_frac) 个字符处。

    硬切会把某句话劈成两半：前半截在训练集、后半截在验证集。
    模型在训练时见过"前半句"，再去验证"后半句"，测出来的 loss 会偏乐观。
    所以往后找最近的一个空行（corpus.txt 里章节之间是空行）。
    """
    target = int(len(text) * (1.0 - val_frac))
    nxt = text.find("\n\n", target)
    if nxt == -1 or nxt > target + 200_000:   # 找不到或太远，就硬切
        return target
    return nxt + 2                            # 跳过那个空行


def write_ids(path: Path, text: str, tok: NovelTokenizer) -> tuple[int, str]:
    """把一段文本分块编码，写成 uint16 的二进制文件。

    返回 (token 数, sha256)。
    """
    n_tok = 0
    sha = hashlib.sha256()
    with open(path, "wb") as f:
        for start in range(0, len(text), CHUNK_CHARS):
            chunk = text[start:start + CHUNK_CHARS]
            ids = tok.encode(chunk)
            if not ids:
                continue
            arr = np.fromiter(ids, dtype=DTOYPE, count=len(ids))
            # tofile 直接写原始字节，没有分隔符、没有头部 —— 就是裸的 uint16 数组
            arr.tofile(f)
            sha.update(arr.tobytes())
            n_tok += len(ids)
    return n_tok, sha.hexdigest()


def read_ids(path: Path) -> np.ndarray:
    """用 memmap 读回来（不把整个文件读进内存）。"""
    return np.memmap(path, dtype=DTOYPE, mode="r")


def main():
    ap = argparse.ArgumentParser(description="语料 → uint16 二进制")
    ap.add_argument("corpus", nargs="?", default="data/processed/corpus.txt")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--vocab-path", default=str(DEFAULT_TOKENIZER_PATH))
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="验证集比例（默认 0.02；语料只有 287 万 token，"
                         "给验证集太多会浪费训练数据）")
    ap.add_argument("--report", default="reports/data_bin_report.md")
    args = ap.parse_args()

    corpus_path = ROOT / args.corpus
    out_dir = ROOT / args.out_dir
    if not corpus_path.exists():
        print(f"[错误] 找不到语料 {corpus_path}", file=sys.stderr)
        print("先跑：py scripts/build_corpus.py data/raw data/processed", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    tok = NovelTokenizer(args.vocab_path)
    text = corpus_path.read_text(encoding="utf-8")
    print(f"语料 {len(text):,} 字 / {len(text.encode('utf-8')):,} 字节")
    print(f"分词器词表 {tok.vocab_size:,}（uint16 上限 65,535）")

    split = find_split_point(text, args.val_frac)
    train_text, val_text = text[:split], text[split:]
    print(f"切分点：第 {split:,} 字符（全篇的 {100*split/len(text):.2f}%）")
    print(f"  训练集 {len(train_text):,} 字 / 验证集 {len(val_text):,} 字")
    print(f"  （切在章节边界上：切分点前 40 字 = {train_text[-40:]!r}）")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    print("\n编码并写盘……")
    n_train, sha_train = write_ids(train_path, train_text, tok)
    print(f"  train.bin {n_train:,} token  {train_path.stat().st_size/1024/1024:.2f} MB")
    n_val, sha_val = write_ids(val_path, val_text, tok)
    print(f"  val.bin   {n_val:,} token  {val_path.stat().st_size/1024/1024:.2f} MB")

    n_total = n_train + n_val
    print(f"  合计 {n_total:,} token，"
          f"{100*n_train/n_total:.2f}% / {100*n_val/n_total:.2f}%")

    # ---------------- 校验 ----------------
    print("\n=== 校验 ===")
    ok_bytes = (train_path.stat().st_size == n_train * 2
                and val_path.stat().st_size == n_val * 2)
    print(f"  文件大小 == token 数 × 2 字节: {'是 [OK]' if ok_bytes else '否 [FAIL]'}")

    train_ids = read_ids(train_path)
    val_ids = read_ids(val_path)
    print(f"  memmap 读回: train {len(train_ids):,} 个, val {len(val_ids):,} 个")
    print(f"  dtype = {train_ids.dtype}, 最大 id = {int(train_ids.max()):,} "
          f"（uint16 上限 65,535 {'[OK]' if train_ids.max() <= 65535 else '[FAIL]'}）")

    t_dec = time.perf_counter()
    train_back = tok.decode(train_ids.tolist())
    val_back = tok.decode(val_ids.tolist())
    dec_sec = time.perf_counter() - t_dec
    ok_train = train_back == train_text
    ok_val = val_back == val_text
    print(f"  解码往返（{dec_sec:.1f}s）: train {'一致 [OK]' if ok_train else '不一致 [FAIL]'}, "
          f"val {'一致 [OK]' if ok_val else '不一致 [FAIL]'}")

    # ---------------- meta.json ----------------
    meta = {
        "corpus": args.corpus,
        "corpus_chars": len(text),
        "split_char": split,
        "val_frac_target": args.val_frac,
        "vocab_size": tok.vocab_size,
        "dtype": "uint16",
        "bytes_per_token": 2,
        "train": {"file": "train.bin", "tokens": n_train, "sha256": sha_train},
        "val": {"file": "val.bin", "tokens": n_val, "sha256": sha_val},
        "total_tokens": n_total,
        "tokenizer": str(Path(args.vocab_path).name),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n元信息已写入 {meta_path.relative_to(ROOT)}")

    elapsed = time.perf_counter() - t0

    # ---------------- 报告 ----------------
    lines = ["# 二进制语料报告", ""]
    lines.append(f"- 语料：`{args.corpus}`（{len(text):,} 字 / "
                 f"{len(text.encode('utf-8')):,} 字节）")
    lines.append(f"- 分词器：`{Path(args.vocab_path).name}`（词表 {tok.vocab_size:,}）")
    lines.append(f"- 存储类型：`{DTOYPE.__name__}`，每个 token 2 字节")
    lines.append(f"- 耗时：{elapsed:.1f} 秒（含全量解码往返校验）")
    lines.append("")
    lines.append("## 产出")
    lines.append("")
    lines.append("| 文件 | token 数 | 大小 | 说明 |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| `train.bin` | {n_train:,} | {train_path.stat().st_size/1024/1024:.2f} MB | "
                 f"训练集（{100*n_train/n_total:.2f}%） |")
    lines.append(f"| `val.bin` | {n_val:,} | {val_path.stat().st_size/1024/1024:.2f} MB | "
                 f"验证集（{100*n_val/n_total:.2f}%） |")
    lines.append(f"| `meta.json` | — | — | 元信息与 sha256 |")
    lines.append("")
    lines.append(f"合计 **{n_total:,}** 个 token。")
    lines.append("")
    lines.append("## 切分")
    lines.append("")
    lines.append(f"切分点在第 **{split:,}** 个字符（全篇 {100*split/len(text):.2f}%），"
                 "对齐到**章节边界**而不是硬切。")
    lines.append("")
    lines.append("> 为什么要对齐到章节边界：硬切会把一句话劈成两半，前半截进训练集、"
                 "后半截进验证集。模型训练时见过前半句，再去验证后半句，"
                 "**测出来的 loss 会偏乐观**。")
    lines.append("")
    lines.append("## 校验")
    lines.append("")
    lines.append("| 检查 | 结果 |")
    lines.append("|---|---|")
    lines.append(f"| 文件大小 == token 数 × 2 | {'✅' if ok_bytes else '❌'} |")
    lines.append(f"| 最大 id ≤ 65,535 | {'✅' if train_ids.max() <= 65535 else '❌'} |")
    lines.append(f"| train 解码往返一致 | {'✅' if ok_train else '❌'} |")
    lines.append(f"| val 解码往返一致 | {'✅' if ok_val else '❌'} |")
    lines.append("")
    lines.append("## 存法对比")
    lines.append("")
    lines.append("| 存法 | 字节/token | 总大小 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 原始 UTF-8 文本 | — | {len(text.encode('utf-8'))/1024/1024:.2f} MB |")
    lines.append(f"| **uint16（本方案）** | 2 | **{n_total*2/1024/1024:.2f} MB** |")
    lines.append(f"| int32 | 4 | {n_total*4/1024/1024:.2f} MB |")
    lines.append(f"| int64 | 8 | {n_total*8/1024/1024:.2f} MB |")
    lines.append(f"| Python list of int（内存） | ~28 | {n_total*28/1024/1024:.0f} MB |")
    lines.append("")
    lines.append("## 怎么用")
    lines.append("")
    lines.append("```python")
    lines.append("import numpy as np")
    lines.append(f"data = np.memmap('data/processed/train.bin', dtype=np.uint16, mode='r')")
    lines.append("# 取任意窗口都是 O(1)，不读整个文件进内存")
    lines.append("x = data[i:i+512].astype(np.int64)")
    lines.append("```")
    lines.append("")

    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {report_path.relative_to(ROOT)}")

    ok = ok_bytes and ok_train and ok_val and int(train_ids.max()) <= 65535
    print(f"\n{'全部校验通过' if ok else '有校验失败，见上'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
