# -*- coding: utf-8 -*-
"""批处理语料构建：data/raw/*.txt → data/processed/{corpus.txt, corpus.jsonl} + 数据报告。

用法:
    py scripts/build_corpus.py data/raw data/processed
    py scripts/build_corpus.py data/raw data/processed --min-chapter-chars 300

设计要点
--------
**单遍流式**：逐本书读进来、清洗、切章、写盘，然后**立刻丢掉正文**。
跨书去重靠的是一个全局的「哈希集合」（只存 32 位十六进制字符串），
所以内存占用与语料总量基本无关——扩到 20 本书也不会炸内存。
反面做法是把所有章节塞进一个 list 再统一处理，书一多就吃光内存。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 让 Windows 控制台正常显示中文（控制台默认 cp936，这里强制 UTF-8 输出）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 让脚本能 import 到 miniqwen 包（脚本在 scripts/ 下，包在上一级）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniqwen.corpus import (  # noqa: E402
    DEDUP_MIN_LEN,
    MAX_TITLE_LEN,
    clean_lines,
    dedup_chapters,
    dedup_paragraphs,
    is_chapter_title,
    iter_chapters,
    read_text,
)

# 中文字符范围（近似）：CJK 统一表意文字 + 扩展A
CJK_RE = None


@dataclass
class BookStat:
    """一本书的处理账本。"""

    file: str
    book: str
    encoding: str
    raw_lines: int
    empty_lines: int
    killed_lines: int
    stripped_lines: int
    residue_lines: int
    kept_lines: int
    chapters_raw: int
    chapters_dup: int
    chapters_out: int
    chars_in: int          # 清洗后、去重前的正文字数
    chars_out: int         # 最终写入的正文字数
    para_dup: int = 0            # 被段落去重删掉的段数
    min_chars_dropped: int = 0   # 因为太短没进 jsonl 的章节数
    glued_titles: int = 0        # 被切开的「标题粘正文」行数
    rule_hits: dict = field(default_factory=dict)   # 每条清洗规则命中多少行

    @property
    def drop_ratio(self) -> float:
        if self.chars_in == 0:
            return 0.0
        return 100.0 * (1 - self.chars_out / self.chars_in)


def find_books(raw_dir: Path) -> list[Path]:
    """找出所有待处理的语料文件，按名字排序保证结果可复现。"""
    files = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".txt", ".text")
    )
    return files


def build_instruction(title: str, genre: str) -> str:
    if title:
        return f"根据标题《{title}》写一篇{genre}小说章节"
    return f"续写下面的{genre}小说内容"


def process_book(
    path: Path,
    ftxt,
    fjson,
    seen_para: set[str],
    seen_chapter: set[str],
    min_chapter_chars: int,
    dedup_min_len: int,
    genre: str,
) -> BookStat:
    """处理一本书：读 → 清洗 → 切章 → 去重 → 写盘。返回账本。

    正文用完即弃：函数返回时不会再持有整本书的内容，
    所以内存占用与语料总量基本无关。
    """
    raw_text, enc = read_text(path)
    clean = clean_lines(raw_text)
    del raw_text          # 原始文本用完即弃，后面只操作清洗结果
    book = clean.book_title or path.stem

    # 「标题粘正文」的脏数据有多少行（真实语料里确实存在）
    glued = sum(
        1 for ln in clean.lines
        if is_chapter_title(ln) and len(ln) > MAX_TITLE_LEN
    )

    chapters_all = list(iter_chapters(clean.lines, book))

    # --- 章节级去重（seen_chapter 由调用方跨书持有）---
    chapters, dup_chapters = dedup_chapters(chapters_all, seen_chapter)

    stat = BookStat(
        file=path.name,
        book=book,
        encoding=enc,
        raw_lines=clean.stats.raw_lines,
        empty_lines=clean.stats.empty_lines,
        killed_lines=clean.stats.killed_lines,
        stripped_lines=clean.stats.stripped_lines,
        residue_lines=clean.stats.residue_lines,
        kept_lines=clean.stats.kept_lines,
        chapters_raw=len(chapters_all),
        chapters_dup=dup_chapters,
        chapters_out=0,
        chars_in=sum(c.chars for c in chapters),
        chars_out=0,
        glued_titles=glued,
        rule_hits=dict(clean.stats.rule_hits),
    )

    for ch in chapters:
        # --- 段落级去重：seen_para 是全局的，跨书生效 ---
        paras, removed = dedup_paragraphs(
            ch.body.split("\n"), dedup_min_len, seen_para
        )
        stat.para_dup += removed
        body = "\n".join(p for p in paras if p)

        if not body:
            continue

        stat.chapters_out += 1
        stat.chars_out += len(body)

        # --- 写 corpus.txt：标题单独一行，章节之间空行 ---
        ftxt.write(f"{ch.title}\n{body}\n\n" if ch.title else f"{body}\n\n")

        # --- 写 corpus.jsonl：太短的章节不进 SFT 数据 ---
        if len(body) >= min_chapter_chars:
            fjson.write(json.dumps(
                {"instruction": build_instruction(ch.title, genre), "output": body},
                ensure_ascii=False,
            ) + "\n")
        else:
            stat.min_chars_dropped += 1

    return stat


def write_report(
    path: Path,
    stats: list[BookStat],
    args: argparse.Namespace,
    elapsed: float,
    paragraphs_dup: int,
    charset: set[str],
    total_chars: int,
    rule_totals: dict[str, int],
) -> None:
    cjk = sum(1 for c in charset if "\u4e00" <= c <= "\u9fff")

    lines: list[str] = []
    lines.append("# 数据报告")
    lines.append("")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 命令：`py scripts/build_corpus.py {args.raw_dir} {args.out_dir}`")
    lines.append(f"- 耗时：{elapsed:.1f} 秒")
    lines.append(f"- 段落去重阈值：{args.dedup_min_len} 字（短于此长度的段落不去重，保护拟声词等文风）")
    lines.append(f"- 章节最低字数：{args.min_chapter_chars} 字（低于此值的章节不进 jsonl）")
    lines.append("")

    lines.append("## 逐本明细")
    lines.append("")
    lines.append("| 文件 | 书名 | 编码 | 原始行 | 空行 | 删噪音行 | 行内切除 | 丢弃残渣 | 章节数 | 去重章 | 正文字数 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        lines.append(
            f"| {s.file} | {s.book} | {s.encoding} | {s.raw_lines} | {s.empty_lines} | "
            f"{s.killed_lines} | {s.stripped_lines} | {s.residue_lines} | "
            f"{s.chapters_raw} | {s.chapters_dup} | {s.chars_in:,} |"
        )
    lines.append("")

    lines.append("## 汇总")
    lines.append("")
    lines.append(f"- 书本数：{len(stats)}")
    lines.append(f"- 总章节数（去重后）：**{sum(s.chapters_out for s in stats):,}**")
    lines.append(f"- 总正文字数（写入 corpus.txt）：**{total_chars:,}**")
    lines.append(f"- 段落去重删除：{paragraphs_dup:,} 段")
    lines.append(f"- 章节去重删除：{sum(s.chapters_dup for s in stats):,} 章")
    lines.append(f"- 因过短未进 jsonl 的章节：{sum(s.min_chars_dropped for s in stats):,} 章")
    lines.append(f"- 字符集大小：{len(charset):,} 种字符，其中中日韩汉字 {cjk:,} 种")
    lines.append(f"- 平均章长：{total_chars / max(1, sum(s.chapters_out for s in stats)):,.0f} 字")
    lines.append("")

    lines.append("## 清洗账本（这本书里删掉的东西都去哪了）")
    lines.append("")
    lines.append("| 环节 | 行数 | 占比 |")
    lines.append("|---|---:|---:|")
    raw_total = sum(s.raw_lines for s in stats) or 1
    for name, val in [
        ("原始总行数", raw_total),
        ("空行（原文一行一段）", sum(s.empty_lines for s in stats)),
        ("整行删除的广告/水印", sum(s.killed_lines for s in stats)),
        ("做过行内切除的行", sum(s.stripped_lines for s in stats)),
        ("切完只剩残渣而丢弃", sum(s.residue_lines for s in stats)),
        ("最终保留的正文行", sum(s.kept_lines for s in stats)),
    ]:
        lines.append(f"| {name} | {val:,} | {100.0 * val / raw_total:.2f}% |")
    lines.append("")

    lines.append("## 清洗规则命中分解")
    lines.append("")
    lines.append("这一节是**自查用**的：某条规则命中数高得离谱时，多半是它在误伤正文。")
    lines.append("实话说，我第一次跑出「行内切除 40706 行」时以为规则炸了，")
    lines.append("拆开才看到 40500 行全是同一个水印词——没有这张表就只能靠猜。")
    lines.append("")
    lines.append("| 规则 | 命中行数 |")
    lines.append("|---|---:|")
    for name, n in sorted(rule_totals.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{name}` | {n:,} |")
    lines.append("")

    lines.append("## 结论与后续")
    lines.append("")
    dup_total = paragraphs_dup + sum(s.chapters_dup for s in stats)
    if dup_total == 0:
        lines.append("- 没有检测到重复内容：单本书通常如此，跨书去重的价值要到 v1（10~20 本）才体现。")
    else:
        lines.append(f"- 去重共处理 {dup_total:,} 处重复。重复率低是正常的：单本书内部本来就不该有大段重复，")
        lines.append("  去重真正的战场是 v1 的多书语料（同一本书被抓两遍、不同站点的转载）。")
    lines.append("- 下一步：用 `scripts/train_tokenizer.py` 在 `corpus.txt` 上训练 BPE 分词器。")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="批处理构建语料")
    ap.add_argument("raw_dir", help="原始语料目录，如 data/raw")
    ap.add_argument("out_dir", help="输出目录，如 data/processed")
    ap.add_argument("--min-chapter-chars", type=int, default=300,
                    help="章节最低字数，低于此值不进 jsonl（默认 300）")
    ap.add_argument("--dedup-min-len", type=int, default=DEDUP_MIN_LEN,
                    help=f"段落去重的长度阈值（默认 {DEDUP_MIN_LEN}）")
    ap.add_argument("--genre", default="玄幻", help="题材，用于拼指令模板")
    ap.add_argument("--report", default="reports/data_report.md", help="报告输出路径")
    args = ap.parse_args(argv)

    t0 = time.time()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.is_dir():
        print(f"[错误] 目录不存在: {raw_dir}", file=sys.stderr)
        return 1

    books = find_books(raw_dir)
    if not books:
        print(f"[错误] {raw_dir} 下没有 .txt 文件", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"发现 {len(books)} 本语料，输出到 {out_dir}")

    txt_path = out_dir / "corpus.txt"
    jsonl_path = out_dir / "corpus.jsonl"

    stats: list[BookStat] = []
    seen_para: set[str] = set()
    seen_chapter: set[str] = set()
    charset: set[str] = set()
    total_chars = 0
    paragraphs_dup = 0
    rule_totals: dict[str, int] = {}

    with open(txt_path, "w", encoding="utf-8") as ftxt, \
         open(jsonl_path, "w", encoding="utf-8") as fjson:
        for i, path in enumerate(books, start=1):
            stat = process_book(
                path, ftxt, fjson, seen_para, seen_chapter,
                args.min_chapter_chars, args.dedup_min_len, args.genre,
            )
            stats.append(stat)
            paragraphs_dup += stat.para_dup
            for k, v in stat.rule_hits.items():
                rule_totals[k] = rule_totals.get(k, 0) + v
            total_chars += stat.chars_out
            print(f"  [{i}/{len(books)}] {path.name} -> 《{stat.book}》 "
                  f"{stat.encoding} {stat.chapters_out} 章 {stat.chars_out:,} 字")

    # 字符集统计：重新扫一遍输出文件（比在内存里攒着省事）
    with open(txt_path, "r", encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(1 << 20), ""):
            charset.update(chunk)

    elapsed = time.time() - t0
    write_report(Path(args.report), stats, args, elapsed,
                 paragraphs_dup, charset, total_chars, rule_totals)

    print("")
    print("=" * 60)
    print(f"corpus.txt   {total_chars:,} 字")
    print(f"corpus.jsonl {sum(1 for _ in open(jsonl_path, encoding='utf-8')):,} 条")
    print(f"数据报告     {args.report}")
    print(f"耗时         {elapsed:.1f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
