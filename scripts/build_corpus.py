# -*- coding: utf-8 -*-
"""批处理语料构建：data/raw/*.txt → data/processed/{corpus.txt, corpus.jsonl} + 数据报告。

用法:
    py scripts/build_corpus.py data/raw data/processed
    py scripts/build_corpus.py data/raw data/processed --min-chapter-chars 300

设计要点
--------
单遍流式：逐本书读进来、清洗、切章、写盘，然后立刻丢掉正文。
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
# [2026-10-13] 知识点：hasattr(obj, "名字") 检查对象身上有没有这个属性，
#              这里是为了兼容老版本 Python（reconfigure 是 3.7 才有的）。
#              try/except/pass = "能切就切，切不了拉倒"——输出乱码只是难看，
#              不值得让程序崩。对比 read_text 里的"编码失败必须 raise"：
#              什么时候宁死报错、什么时候优雅降级，看失败的代价。
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception: 
        pass

# 让脚本能 import 到 miniqwen 包（脚本在 scripts/ 下，包在上一级）
# [2026-10-13] 知识点：.parent.parent 跳两级到项目根。sys.path 是"去哪找模块"
#              的清单，insert(0, ...) 插到最前面 = 优先级最高。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# [2026-10-13] 知识点：from 包 import 具体名字 大体相当于 C++ 的
#              using corpus::clean_lines; 按名取件，命名空间不污染。
#              不用 import *（≈ using namespace），那会把别人柜子里的东西
#              全泼进来，撞名了都不知道。
#              一个区别：取函数/类时指向同一个对象，当 using 用没问题；
#              取【变量】时是"拷贝绑定"——模块之后重新赋值，你手上那个名字
#              还指向旧对象，而 C++ 的 using 是别名，会跟着变。
# [2026-10-13] 知识点：# noqa: E402 是给 linter 看的暗号——"我知道 import 没在
#              文件顶部，是故意的（要先插 sys.path），别报警"。
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


# [2026-10-13] 知识点：@dataclass 自动生成 __init__/__repr__/__eq__，只写字段就行。
#              rule_hits 用 field(default_factory=dict)：可变默认值必须现造，
#              直接写 = {} 会让所有实例共享同一个字典（a 记账 b 也变）。
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

    # [2026-10-13] 知识点：@property 让方法用起来像属性（s.drop_ratio 不加括号），
    #              每次访问现场算，不存可能过期的冗余数据。这里还顺手防了除零。
    @property
    def drop_ratio(self) -> float:
        if self.chars_in == 0:
            return 0.0
        return 100.0 * (1 - self.chars_out / self.chars_in)


def find_books(raw_dir: Path) -> list[Path]:
    """找出所有待处理的语料文件，按名字排序保证结果可复现。"""
    # [2026-10-13] 知识点：生成器表达式 (x for x in ... if 条件) 直接喂给 sorted，
    #              不用先建中间列表。排序是为了可复现——目录遍历顺序在不同系统上
    #              不固定，排了序每次运行的处理顺序才一样。
    # [2026-10-13] 知识点：p.suffix 是扩展名（带点，如 ".txt"），p.stem 是砍掉
    #              扩展名的主干。.lower() 防 ".TXT" 这种大写漏网。
    files = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".txt", ".text")
    )
    return files


def build_instruction(title: str, genre: str) -> str:
    # [2026-10-13] 知识点：if title 利用"空字符串布尔值为 False"做判断；
    #              f-string 里 {变量} 直接插值。genre 参数化让流水线换本书
    #              换个题材就能复用。
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
    # [2026-10-13] 知识点：del 主动删引用，让这块大内存尽早被垃圾回收。
    #              一般不用管（Python 自动回收），但大文件流水线里是好习惯。
    del raw_text          # 原始文本用完即弃，后面只操作清洗结果

    # [2026-10-13] 知识点：A or B 兜底写法——A 是假值（None/""）就取 B。
    #              这里：扫到书名用书名，没扫到拿文件名（path.stem）顶上。
    #              注意坑：合法值本身可能是 0/"" 时不能用 or，要用
    #              x if x is not None else 兜底。
    book = clean.book_title or path.stem

    # 「标题粘正文」的脏数据有多少行（真实语料里确实存在）
    # [2026-10-13] 知识点：sum(1 for x in ... if 条件) = "数有多少个满足条件"，
    #              一行顶四行循环，Python 计数的标准套路。
    glued = sum(
        1 for ln in clean.lines
        if is_chapter_title(ln) and len(ln) > MAX_TITLE_LEN
    )

    # iter_chapters 是生成器（yield），list(...) 把它产出的章节一次收齐
    chapters_all = list(iter_chapters(clean.lines, book))

    # --- 章节级去重（seen_chapter 由调用方跨书持有）---
    # 在 main 里创建一次、逐本传进来，20 本书共用一个登记处，
    # 跨书重复才抓得到。函数内部自己 set() 就降级成单本去重了。
    chapters, dup_chapters = dedup_chapters(chapters_all, seen_chapter)

    # [2026-10-13] 知识点：dataclass 生成的 __init__ 支持全关键字传参，
    #              十几个字段也不会搞混顺序——比位置传参可读得多。
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
        # join 是 split 的反操作：列表 → 用 \n 粘回字符串。
        # p for p in paras if p：顺手过滤空段落（空串布尔值是 False）。
        body = "\n".join(p for p in paras if p)

        # 整章段落全被去重删光了 → 空壳章节，跳过（双保险）
        if not body:
            continue

        stat.chapters_out += 1
        stat.chars_out += len(body)

        # --- 写 corpus.txt：标题单独一行，章节之间空行 ---
        # [2026-10-13] 知识点：三元表达式 A if 条件 else B——有标题写标题行，
        #              没标题（前言）不写，免得留个空行在开头。
        ftxt.write(f"{ch.title}\n{body}\n\n" if ch.title else f"{body}\n\n")

        # --- 写 corpus.jsonl：太短的章节不进 SFT 数据 ---
        # [2026-10-13] 知识点：json.dumps 把字典序列化成 JSON 字符串；
        #              ensure_ascii=False 保留汉字（默认会转义成 \u6597 这种）。
        #              每行一条 JSON = jsonl 格式，可以逐行流式读。
        #              双格式分流：txt 全收（tokenizer 要全），jsonl 设门槛
        #              （微调要精，短章没训练价值）。
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
    # [2026-10-13] 知识点：这个函数不是写 log，是写"数据质量验收单"——
    #              log 记过程（边跑边写），report 汇总结论（跑完一次生成）。
    #              原则：可以丢数据，但必须知道丢了多少、为什么丢。

    # [2026-10-13] 知识点：链式比较是 Python 特有写法，a <= c <= b 等价于
    #              a <= c and c <= b。（C++ 里 a <= c <= b 会算成 (a<=c)<=b，
    #              结果恒为 true/false，是经典错误。）
    #              这里的 \u4e00 和 \u9fff 是转义写法，指向汉字 Unicode 范围
    #              U+4E00~U+9FFF；直接写字面汉字 "一" <= c <= "鿿" 也完全等价，
    #              转义形式的好处是不受源文件编码影响。
    cjk = sum(1 for c in charset if "\u4e00" <= c <= "\u9fff")

    # 攒列表再一次写盘：几十次小 IO 合并成一次，比一行行 write 快
    lines: list[str] = []
    lines.append("# 数据报告")
    lines.append("")
    # [2026-10-13] 知识点：f-string 格式语法 {x:格式}——
    #              :.1f 保留 1 位小数；:, 千分位逗号；:,.0f 两个叠加。
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 命令：`py scripts/build_corpus.py {args.raw_dir} {args.out_dir}`")
    lines.append(f"- 耗时：{elapsed:.1f} 秒")
    lines.append(f"- 段落去重阈值：{args.dedup_min_len} 字（短于此长度的段落不去重，保护拟声词等文风）")
    lines.append(f"- 章节最低字数：{args.min_chapter_chars} 字（低于此值的章节不进 jsonl）")
    lines.append("")

    lines.append("## 逐本明细")
    lines.append("")
    # Markdown 表格：第一行表头，第二行分隔（---: 该列右对齐，数字列的习惯）
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
    # [2026-10-13] 知识点：sum(x.字段 for x in list)——生成器直接喂 sum，
    #              "把所有书的某字段加起来"一行完成。
    lines.append(f"- 总章节数（去重后）：**{sum(s.chapters_out for s in stats):,}**")
    lines.append(f"- 总正文字数（写入 corpus.txt）：**{total_chars:,}**")
    lines.append(f"- 段落去重删除：{paragraphs_dup:,} 段")
    lines.append(f"- 章节去重删除：{sum(s.chapters_dup for s in stats):,} 章")
    lines.append(f"- 因过短未进 jsonl 的章节：{sum(s.min_chars_dropped for s in stats):,} 章")
    lines.append(f"- 字符集大小：{len(charset):,} 种字符，其中中日韩汉字 {cjk:,} 种")
    # max(1, ...) 防除零：一章都没产出时分母保底 1
    lines.append(f"- 平均章长：{total_chars / max(1, sum(s.chapters_out for s in stats)):,.0f} 字")
    lines.append("")

    lines.append("## 清洗账本（这本书里删掉的东西都去哪了）")
    lines.append("")
    lines.append("| 环节 | 行数 | 占比 |")
    lines.append("|---|---:|---:|")
    # or 1 兜底：总行数为 0 时防除零
    raw_total = sum(s.raw_lines for s in stats) or 1
    # [2026-10-13] 知识点：数据驱动循环——把 (名字, 数值) 放进元组列表再 for，
    #              加一行数据只改列表，循环体不动。数据和逻辑分离。
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
    # [2026-10-13] 知识点：sorted(d.items(), key=lambda kv: -kv[1])
    #              d.items() 拿 (键,值) 对；key 指定排序依据；
    #              lambda kv: -kv[1] 按命中数取负 → 从大到小排，
    #              命中最多的规则排最前面，一眼看到"最狠的规则"。
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

    # [2026-10-13] 知识点：mkdir(parents=True, exist_ok=True)——parents 递归建
    #              多级目录，exist_ok 已存在不报错，两个参数缺一不可。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    # [2026-10-13] 知识点：argparse 是 sys.argv 的正经替代——
    #              "raw_dir"（不带--）是位置参数，必填；
    #              "--xxx" 是选项参数，不给就用 default；type=int 自动转换；
    #              白送 --help 文档。参数超过两个就该用它。
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

    # [2026-10-13] 知识点：file=sys.stderr 让报错走"标准错误"通道，
    #              不混进 stdout 的正常输出（别人 > 重定向时错误不会污染结果）。
    #              配合 return 1（失败退出码），调度脚本能判断成败。
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

    # 跨书去重的心脏：两个指纹集合在循环【外面】创建，
    # 所有书共用同一个登记处。挪进 process_book 内部就失效了。
    stats: list[BookStat] = []
    seen_para: set[str] = set()
    seen_chapter: set[str] = set()
    charset: set[str] = set()
    total_chars = 0
    paragraphs_dup = 0
    rule_totals: dict[str, int] = {}

    # [2026-10-13] 知识点：with open(...) as f ≈ 文件版 RAII，缩进块结束自动
    #              关文件（抛异常也关）；逗号连写管多个文件；\ 是续行符。
    #              两个输出文件全程开着，20 本书的内容追加进同一对文件。
    # [2026-10-13] 知识点：enumerate(books, start=1) 同时拿序号和元素，
    #              start=1 让序号从 1 开始（打 [3/20] 进度用）。
    with open(txt_path, "w", encoding="utf-8") as ftxt, \
         open(jsonl_path, "w", encoding="utf-8") as fjson:
        for i, path in enumerate(books, start=1):
            stat = process_book(
                path, ftxt, fjson, seen_para, seen_chapter,
                args.min_chapter_chars, args.dedup_min_len, args.genre,
            )
            stats.append(stat)
            paragraphs_dup += stat.para_dup
            # [2026-10-13] 知识点：d.items() 遍历字典键值对；
            #              d[k] = d.get(k, 0) + v 是累加计数的标准套路
            #              （get 在键不存在时返回 0，不会 KeyError）。
            for k, v in stat.rule_hits.items():
                rule_totals[k] = rule_totals.get(k, 0) + v
            total_chars += stat.chars_out
            print(f"  [{i}/{len(books)}] {path.name} -> 《{stat.book}》 "
                  f"{stat.encoding} {stat.chapters_out} 章 {stat.chars_out:,} 字")

    # 字符集统计：重新扫一遍输出文件（比在内存里攒着省事）
    # [2026-10-13] 知识点：iter(函数, 哨兵) 反复调用直到返回哨兵值——
    #              每次读 1MB（1 << 20 = 2^20，位移运算），读到空串（文件尾）停。
    #              lambda: ... 是一次性匿名函数。分块读，不一次塞爆内存。
    #              这个数决定 tokenizer 初始词表该多大。
    with open(txt_path, "r", encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(1 << 20), ""):
            charset.update(chunk)

    elapsed = time.time() - t0
    write_report(Path(args.report), stats, args, elapsed,
                 paragraphs_dup, charset, total_chars, rule_totals)

    print("")
    print("=" * 60)
    # [2026-10-13] 知识点：sum(1 for _ in open(...)) 数文件行数；
    #              _ 是惯例变量名，表示"这值我不用，只关心循环次数"。
    print(f"corpus.txt   {total_chars:,} 字")
    print(f"corpus.jsonl {sum(1 for _ in open(jsonl_path, encoding='utf-8')):,} 条")
    print(f"数据报告     {args.report}")
    print(f"耗时         {elapsed:.1f} 秒")
    return 0


# [2026-10-13] 知识点：__name__ 是 Python 自动设置的变量——直接运行时是
#              "__main__"，被 import 时是模块名。这个 if 保证 main 只在
#              "当脚本跑"时执行，被 import 时不会误触发。
# [2026-10-13] 知识点：sys.exit(main()) 把 main 的返回值当退出码交给系统
#              （0 成功 / 非 0 失败）≈ C++ main 的 return 0。
#              裸写 main() 会丢弃返回值，失败了系统也以为成功。
if __name__ == "__main__":
    sys.exit(main())
