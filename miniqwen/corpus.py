# -*- coding: utf-8 -*-
"""语料处理核心库：编码探测 → 清洗 → 切章 → 去重。

设计原则
--------
1. **只做纯函数**：不读 sys.argv、不 print、不写文件。
   所以每个环节都能被 pytest 单独测，也能被别的脚本复用。
2. **规则集中在文件顶部**：改规则不用翻代码正文。
3. **每条规则都要有真实数据支撑**：规则下面的注释写的是它在
   《斗破苍穹》10.8MB 真实语料上的侦察结论，不是拍脑袋定的。

用法:
    from miniqwen.corpus import read_text, clean_lines, iter_chapters

    text, enc = read_text("data/raw/某小说.txt")
    result = clean_lines(text)
    for ch in iter_chapters(result.lines):
        print(ch.title, len(ch.body))
"""

from __future__ import annotations

import codecs
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# ============================================================
# 规则区
# ============================================================

# 没有 BOM 时按这个顺序试。顺序的依据是**实测**，不是感觉：
#   - utf-8 校验严格：用 UTF-8 去解 GBK 中文，20000 行真实语料里 0 行成功（100% 抛异常）
#   - gbk 也远比想象的严格：用 GBK 去解 UTF-8 中文，20000 行里只有 7 行"成功"（0.03%）
#     ⚠️ 我原先在这里写过"gbk 几乎能成功解码任何字节流"——实验证明是错的，已更正。
#   - utf-16 **不能放在这里**：Python 的 utf-16 在没有 BOM 时会退化成 LE 硬解，
#     而不是报错。所以 utf-16 必须靠 BOM 显式判断（见 read_text）。
NO_BOM_FALLBACKS: tuple[str, ...] = ("utf-8", "gbk")

# 书名行，如 《斗破苍穹》。它在正文之前单独成行，用来给这本书起名。
BOOK_TITLE_RE = re.compile(r"^《([^》]{1,60})》$")

# 章节标题。真实语料里 1442 个「第X章」，另有「第X回/节/卷」和「楔子/序章」等写法。
#
# 「第X章」后面允许直接跟标题名（"第一章 陨落的天才"）甚至粘正文（脏数据），
# 所以不能用「必须跟空白」来约束它。
#
# 但「楔子/序章/尾声/后记/番外」这类词**必须**用零宽断言卡住后边界：
# 否则正文里一句「序章的内容写在这里」会被当成章节标题，
# 结果真正的序章正文变成空章节被丢掉——这个 bug 是测试样例抓出来的。
_ORDINAL_MARKER = r"第[零一二三四五六七八九十百千万两0-9]+[章回节卷]"
_WORD_MARKER = r"(?:楔子|序[章言]|尾声|后记|番外)(?=[\s：:、，,.．。\-—]|$)"
CHAPTER_MARKER = rf"{_ORDINAL_MARKER}|{_WORD_MARKER}"
CHAPTER_RE = re.compile(rf"^({CHAPTER_MARKER})")

# 标题最长多少字。真实语料里有「第七百七十六章狼狈的铁护法...」这种
# 标题和正文粘在一行的脏数据（最长 60+ 字），超过这个长度就要切开。
MAX_TITLE_LEN = 30

# 句子结束标点，用来在粘连的长行里找切点。
SENTENCE_END = "，。！？；：、,.!?;:"

# 整行删除：整行内容基本就是广告/水印，没有任何正文价值。
# 注意：这里绝不能只写「小说」二字——真实语料里有
# 「大陆上并没有小说中常见的各系魔法」这种合法正文。
LINE_KILL_PATTERNS: tuple[str, ...] = (
    r"^[（(]?未完待续[）)]?$",
    r"^[（(]?(?:求|跪求)(?:收藏|推荐|月票|订阅)[）)]?$",
    r"^(?:请收藏本站|请记住本站|最新章节|全文阅读|手机阅读)$",
    r"^(?:16k|www\.|https?://)",
    r"^[＝=＊*\-—_]{4,}$",            # 分隔线
)

# 行内局部切除：真实语料里水印**全部藏在行中间**，例如
#   「…继续研习着手中的碎石掌。16k小说wαp.16k整理」
#   「"谷尼叔叔…"小说整理发布于16k"略微沉默了一会…」
# 而且盗版站会用全角字符伪装（wαp 里的 α 是希腊字母、wàp 里的 à 是重音字母），
# 所以只写 wap\. 会全部漏掉。
#
# ⚠️ URL 后面绝不能用 \S* ——中文没有空格，\S* 会一路吃到行尾，
# 把紧跟其后的**整段正文**一起删掉。所以限定成 URL 合法字符集 + 长度上限。
_URL_TAIL = r"[A-Za-z0-9./\-_?=&%~#:+,]{0,80}"
INLINE_STRIP_PATTERNS: tuple[str, ...] = (
    r"https?://" + _URL_TAIL,
    r"www\.[A-Za-z0-9.\-]+\.(?:com|net|org|cn|cc|top|xyz|net)" + _URL_TAIL,
    r"[A-Za-z0-9.\-]{2,}\.(?:com|net|org|cn|cc|top|xyz)" + _URL_TAIL,
    r"w[αàáaä]p[.．][^\s，。！？；”]{0,20}",      # wap./wαp./wàp. 全角伪装
    r"16k[^\s，。！？；]{0,20}",
    r"全文字版小说阅读[^。！？]{0,40}",
    r"小说整理发布于[^\s，。！？；”]{0,12}",
    r"小说整理[^\s，。！？；”]{0,12}",
    r"[（(]未完待续[）)]",
    r"[（(]?未完待续[，,][^）)。！？]{0,60}[）)]?",   # (未完待续，如欲知后事如何…)
    r"【[^】]{0,20}(?:网|小说|阅读|下载|收藏|更新)[^】]{0,20}】",
    r"[（(][^）)]{0,20}(?:网|小说|阅读|下载|收藏|更新)[^）)]{0,20}[）)]",
)

# 反盗版水印词典：盗版站会把**别的小说名**插进正文当追踪标记。
# 《斗破苍穹》语料实测：另一个书名「武动乾坤」被插入了 **40500 次**，
# 覆盖了一半以上的行（同一作者的另一本书名，盗版站的追踪水印）。
# 这类词没有通用规律（它看起来就是正常中文），只能维护词典——
# 这是数据工程里很现实的一幕：没有银弹，就是攒规则、攒词表。
WATERMARK_WORDS: tuple[str, ...] = (
    "武动乾坤",
    "大主宰",
)

# 段落去重的长度阈值。
# 侦察结论：重复最多的短段落是「"嘭！"205次」「"轰！"142次」——
# 它们是**文风**不是垃圾，无差别去重会把拟声词删掉。
# 所以只对长度 >= 这个值的段落去重。
DEDUP_MIN_LEN = 20

# 行内切除后判定为「残渣」的门槛：
# 原行超过 KEEP_IF_STRIPPED_FROM 字、切完剩不到 MIN_KEPT_AFTER_STRIP 字 → 丢弃。
# 修复点：原版 clean_novel.py 写的是 `if len(line) >= 4`，
# 会把所有短段落（"轰！"、"嘭！"）一起删掉——那是**正文**，不是噪音。
KEEP_IF_STRIPPED_FROM = 8
MIN_KEPT_AFTER_STRIP = 4


# ============================================================
# 数据结构
# ============================================================
@dataclass
class CleanStats:
    """一次清洗各环节删掉了多少东西。用于数据报告，也用于自检。"""

    raw_lines: int = 0
    empty_lines: int = 0
    killed_lines: int = 0        # 整行删除的噪音行
    stripped_lines: int = 0      # 做了行内切除的行
    residue_lines: int = 0       # 切完只剩残渣而丢弃的行
    kept_lines: int = 0
    # 每条规则各命中多少行。**这个必须记录**：
    # 报告里「行内切除 40706 行」曾让我以为规则误伤，
    # 拆开一看 40500 行全是同一个水印词——没有这个分解表就只能靠猜。
    rule_hits: dict[str, int] = field(default_factory=dict)


@dataclass
class CleanResult:
    lines: list[str] = field(default_factory=list)
    stats: CleanStats = field(default_factory=CleanStats)
    book_title: str | None = None


@dataclass
class Chapter:
    title: str
    body: str
    book: str = ""

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}" if self.title else self.body

    @property
    def chars(self) -> int:
        return len(self.body)


# ============================================================
# 1. 读文件
# ============================================================
def read_text(path: str | Path) -> tuple[str, str]:
    """读文件并自动识别编码，返回 (文本, 编码名)。

    策略（先看证据，再看猜测）：
      1. **有 BOM 就按 BOM 解**——这是文件自己声明的编码，不用猜。BOM 还顺便被去掉，
         否则 \\ufeff 会混进正文变成一个字表里的怪 token。
      2. 没有 BOM 时，依次试 NO_BOM_FALLBACKS（utf-8 严格，然后 gbk）。
      3. 全都不行就报错，**绝不静默降级**（errors="replace" 会掩盖问题）。

    为什么不能只用「依次 try/except」：Python 的 utf-16 在没有 BOM 时不会报错，
    而是按小端硬解。一个短文件万一"解成功"就是静默乱码，查都查不出来。
    """
    raw = Path(path).read_bytes()

    # 1) BOM 优先
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16"), "utf-16"

    # 2) 无 BOM：按实测过的顺序试
    for enc in NO_BOM_FALLBACKS:
        try:
            return raw.decode(enc), enc
        except UnicodeError:
            continue

    raise ValueError(
        f"无法识别编码: {path}（试过 utf-8-sig/utf-16(BOM) 和 {', '.join(NO_BOM_FALLBACKS)}）"
    )


# ============================================================
# 2. 清洗
# ============================================================
def strip_inline(line: str, rule_hits: dict[str, int] | None = None) -> str:
    """行内局部切除：把水印从正文里挖掉，保留其余正文。

    传 rule_hits 可以统计每条规则命中了多少行，用于数据报告和自查。
    """
    for pat in INLINE_STRIP_PATTERNS:
        new = re.sub(pat, "", line)
        if new != line and rule_hits is not None:
            rule_hits[pat] = rule_hits.get(pat, 0) + 1
        line = new

    for word in WATERMARK_WORDS:
        if word in line:
            if rule_hits is not None:
                key = f"水印词典:{word}"
                rule_hits[key] = rule_hits.get(key, 0) + 1
            line = line.replace(word, "")

    # 挖掉 URL 后常常留下连续空格（"详情请访问 www.x.com 获取" → "详情请访问  获取"），
    # 合并成一个，别让它变成两个 token。
    line = re.sub(r"[ \t\u3000]{2,}", " ", line)
    return line.strip()


def is_noise_line(line: str) -> bool:
    """整行是不是纯噪音（没有任何正文价值）。"""
    return any(re.search(pat, line) for pat in LINE_KILL_PATTERNS)


def clean_lines(text: str) -> CleanResult:
    """三层清洗：整行删除 → 行内切除 → 残渣丢弃。

    返回 CleanResult（干净的行 + 统计 + 书名）。
    """
    result = CleanResult()
    stats = result.stats

    raw_lines = text.splitlines()
    stats.raw_lines = len(raw_lines)

    for raw in raw_lines:
        line = raw.strip()

        # --- 空行：真实语料里 66% 的行是空行（原文"一行一段"）---
        if not line:
            stats.empty_lines += 1
            continue

        # --- 书名行：抽出来当书名，不进正文 ---
        m = BOOK_TITLE_RE.match(line)
        if m and result.book_title is None and not result.lines:
            result.book_title = m.group(1)
            continue

        # --- 第一层：整行删除 ---
        if is_noise_line(line):
            stats.killed_lines += 1
            continue

        # --- 第二层：行内局部切除 ---
        before_len = len(line)
        cleaned = strip_inline(line, stats.rule_hits)
        after_len = len(cleaned)

        if after_len != before_len:
            stats.stripped_lines += 1

        # --- 第三层：只有当「原来挺长、切完只剩渣」时才丢弃 ---
        if before_len >= KEEP_IF_STRIPPED_FROM and after_len < MIN_KEPT_AFTER_STRIP:
            stats.residue_lines += 1
            continue

        # 空掉的行丢弃；剩下的**不管多短都保留**（"轰！"是正文）
        if not cleaned:
            stats.empty_lines += 1
            continue

        result.lines.append(cleaned)

    stats.kept_lines = len(result.lines)
    return result


# ============================================================
# 3. 切章
# ============================================================
def is_chapter_title(line: str) -> bool:
    return bool(CHAPTER_RE.match(line))


def split_glued_title(line: str) -> tuple[str, str]:
    """把「标题粘正文」的长行切回 (标题, 正文)。

    真实脏数据样例：
      「第七百七十六章狼狈的铁护法的铁护法帝遇，据薰儿所说，这是其族中秘技之一…」
    标题本该只有「第七百七十六章狼狈的铁护法」，后面的正文被粘在了同一行。

    切法：在 MAX_TITLE_LEN 字以内找第一个句子标点，从那里断开；
    找不到就硬切。切不干净是正常的——这是启发式，不是解析器。
    """
    if len(line) <= MAX_TITLE_LEN:
        return line, ""

    cut = -1
    for i, ch in enumerate(line[:MAX_TITLE_LEN], start=1):
        if ch in SENTENCE_END:
            cut = i
            break
    if cut < 0:
        cut = MAX_TITLE_LEN

    return line[:cut].rstrip(SENTENCE_END + " "), line[cut:].strip()


def iter_chapters(lines: Sequence[str], book: str = "") -> Iterator[Chapter]:
    """把清洗后的行流切成章节。

    - 第一个章标题之前的内容如果非空，作为「前言」保留（title 为空）。
    - 正文为空的章节不产出（避免"只有一个标题没有内容"的脏章节污染数据）。
    """
    pending: list[str] = []
    title: str | None = None

    for line in lines:
        if is_chapter_title(line):
            body = "\n".join(pending).strip()
            if body:
                yield Chapter(title or "", body, book)
            title, glued = split_glued_title(line)
            pending = [glued] if glued else []
        else:
            pending.append(line)

    body = "\n".join(pending).strip()
    if body:
        yield Chapter(title or "", body, book)


# ============================================================
# 4. 去重
# ============================================================
def _norm(s: str) -> str:
    """归一化：去掉空白和标点，只留文字。用于判断「内容是否一样」。"""
    return re.sub(r"[\s\W_]+", "", s)


def dedup_paragraphs(
    paragraphs: Iterable[str],
    min_len: int = DEDUP_MIN_LEN,
    seen: set[str] | None = None,
) -> tuple[list[str], int]:
    """段落级精确去重。

    只对长度 >= min_len 的段落去重：短段落（拟声词、口头禅）是文风。
    返回 (保留的段落, 删掉的段落数)。

    `seen` 必须由调用方持有并跨书传进来——否则每本书都新建一个集合，
    跨书去重就完全失效了（这是个很容易踩的坑：函数内部自己 set() 看起来
    很干净，实际把「全局去重」偷偷降级成了「单本去重」）。
    """
    if seen is None:
        seen = set()
    kept: list[str] = []
    removed = 0

    for p in paragraphs:
        if len(p) < min_len:
            kept.append(p)
            continue
        h = hashlib.md5(p.encode("utf-8")).hexdigest()
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        kept.append(p)

    return kept, removed


def dedup_chapters(
    chapters: Iterable[Chapter], seen: set[str] | None = None
) -> tuple[list[Chapter], int]:
    """章节级去重：**精确重复**的章节（正文归一化后完全一致）。

    用归一化后的正文做指纹，能吃掉「只差空白/标点/编码」的重复章节。

    ⚠️ 已知边界（有意不做的）：如果两个源在正文里插入了不同的水印
    （如甲源是「正文」，乙源是「正文 16k小说」），归一化后仍然不同，
    这个函数**抓不到**。要抓这种需要 SimHash / MinHash 近似去重，
    代价是每章要存一份 shingle 指纹、且 Python 里算得慢。
    单本书语料里没有这种场景（侦察实测重复率上限 0.43%），
    所以 v1 扩到多书语料时再实现，别提前优化。
    """
    if seen is None:
        seen = set()
    kept: list[Chapter] = []
    removed = 0

    for ch in chapters:
        norm = _norm(ch.body)
        if len(norm) < 50:            # 太短的章节不做去重，避免误杀
            kept.append(ch)
            continue
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        kept.append(ch)

    return kept, removed
