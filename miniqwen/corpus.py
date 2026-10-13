# -*- coding: utf-8 -*-
"""语料处理核心库：编码探测 → 清洗 → 切章 → 去重。

约定：
1. 只做纯函数：不读 sys.argv、不 print、不写文件。
   这样每个环节都能被 pytest 单独测，也能被别的脚本复用。
2. 规则全部放在文件顶部，改规则不用翻正文。
3. 每条规则下面的注释写的是它在《斗破苍穹》10.8MB 真实语料上的
   侦察结论，不是拍脑袋定的。

用法:
    from miniqwen.corpus import read_text, clean_lines, iter_chapters

    text, enc = read_text("data/raw/某小说.txt")
    result = clean_lines(text)
    for ch in iter_chapters(result.lines):
        print(ch.title, len(ch.body))
"""

# [2026-10-12] 知识点：from __future__ import annotations 让本文件所有类型注解
#              变成"延迟求值"（原样存着不求值），于是注解里可以写还没定义的名字，
#              也不用为了前向引用加引号。它必须放在文件最前面：
#              只能排在文档字符串和注释之后、其他 import 之前。
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

# [2026-10-12] 知识点：tuple[str, ...] 里的 ... 表示"任意长度、每个都是 str"，
#              区别于 tuple[str, str]（恰好两个）。str 或 None 写成 str | None。
NO_BOM_FALLBACKS: tuple[str, ...] = ("utf-8", "gbk")

# 书名行，如 《斗破苍穹》。它单独成行出现在正文之前，用来给这本书起名。
# [2026-10-12] 知识点：re.compile 把规则字符串预编译成 Pattern 对象，编译一次
#              全局复用。正则【语法】写错会在 import 时就抛 re.error；
#              但"规则太宽、误伤正文"这种逻辑错误编译期发现不了，只能靠命中统计查。
# [2026-10-12] 知识点：正则字符串前面加 r（raw string），防止 \n \t 这类
#              反斜杠组合被 Python 字符串层先吃掉。
BOOK_TITLE_RE = re.compile(r"^《([^》]{1,60})》$")

# 章节标题。真实语料里 1442 个「第X章」，另有「第X回/节/卷」和「楔子/序章」等。
#
# 「第X章」后面允许直接跟标题名（"第一章 陨落的天才"）甚至粘正文（脏数据），
# 所以不能用「必须跟空白」来约束它。
#
# 但「楔子/序章/尾声/后记/番外」这类词必须用零宽断言卡住后边界，
# 否则正文里一句「序章的内容写在这里」会被当成章节标题，
# 真正的序章正文会变成空章节被丢掉
#
# [2026-10-12] 知识点：[...] 每次只匹配一个字符，"1083" 能匹配是因为 + 把
#              "选一个"的动作重复了多次。[^》] 表示"除》外任意字符"。
# [2026-10-12] 知识点：(?:...) 只分组不捕获（不占 group 编号）；
#              (?=...) 是前瞻断言，向右看一眼但不消耗字符。
# [2026-10-12] 知识点：rf"..." 是 f-string 插值和 raw string 叠加，
#              把两条规则用 | 拼成一条，避免写一坨没人读得动的长正则。
_ORDINAL_MARKER = r"第[零一二三四五六七八九十百千万两0-9]+[章回节卷]"
_WORD_MARKER = r"(?:楔子|序[章言]|尾声|后记|番外)(?=[\s：:、，,.．。\-—]|$)"
CHAPTER_MARKER = rf"{_ORDINAL_MARKER}|{_WORD_MARKER}"
CHAPTER_RE = re.compile(rf"^({CHAPTER_MARKER})")

# 标题最长多少字。真实语料里有「第七百七十六章狼狈的铁护法...」这种
# 标题和正文粘在一行的脏数据（最长 30+ 字），超过这个长度就要切开。
MAX_TITLE_LEN = 30

# 句子结束标点，用来在粘连的长行里找切点。
SENTENCE_END = "，。！？；：、,.!?;:"

# 整行删除：整行基本就是广告/水印，没有任何正文价值。
# 真实语料里有
# 「大陆上并没有小说中常见的各系魔法」这种合法正文。
# [2026-10-12] 知识点：[（(]? 里的 ? 修饰左边的字符类，表示括号可有可无，
#              数量词永远跟在"被修饰的东西"后面。
LINE_KILL_PATTERNS: tuple[str, ...] = (
    r"^[（(]?未完待续[）)]?$",
    r"^[（(]?(?:求|跪求)(?:收藏|推荐|月票|订阅)[）)]?$",
    r"^(?:请收藏本站|请记住本站|最新章节|全文阅读|手机阅读)$",
    r"^(?:16k|www\.|https?://)",
    r"^[＝=＊*\-—_]{4,}$",            # 分隔线
)

# 行内局部切除：真实语料里水印全部藏在行中间，例如
#   「…继续研习着手中的碎石掌。16k小说wαp.16k整理」
#   「"谷尼叔叔…"小说整理发布于16k"略微沉默了一会…」
# 盗版站会用全角字符伪装（wαp 的 α 是希腊字母、wàp 的 à 是重音字母），
# 只写 wap\. 会全部漏掉。
#
# 注意：URL 后面绝不能用 \S*——中文没有空格，\S* 会一路吃到行尾，
# 把紧跟其后的整段正文一起删掉。所以限定成 URL 合法字符集 + 长度上限。
# [2026-10-12] 知识点：字符类里的 - 夹在两个字符中间时表示范围（A-Z），
#              要匹配字面的减号必须写成 \-。
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

# 反盗版水印词典：盗版站会把别的小说名插进正文当追踪标记。
# 《斗破苍穹》实测：「武动乾坤」被插了 40500 次，覆盖一半以上的行
# （同一作者的另一本书名）。这类词看起来就是正常中文，没有通用规律，
# 只能攒词典。
WATERMARK_WORDS: tuple[str, ...] = (
    "武动乾坤",
    "大主宰",
)

# 段落去重的长度阈值。
# 第一版中重复最多的短段落是「"嘭！"205次」「"轰！"142次」
# 它们是文风不是垃圾，无差别去重会把拟声词删掉。所以只去重长段落。
DEDUP_MIN_LEN = 20

# 行内切除后判定为「残渣」的门槛：
# 原行超过 KEEP_IF_STRIPPED_FROM 字、切完剩不到 MIN_KEPT_AFTER_STRIP 字 → 丢。
# 修复点：旧版 clean_novel.py 写的是 if len(line) >= 4，
# 会把所有短段落（"轰！"、"嘭！"）一起删掉
KEEP_IF_STRIPPED_FROM = 8
MIN_KEPT_AFTER_STRIP = 4


# ============================================================
# 数据结构
# ============================================================
# [2026-10-12] 知识点：@dataclass 自动帮我生成 __init__/__repr__/__eq__，
#              只写字段就行。__init__ 这类双下划线名字是 Python 的约定接口，
#              名字固定、改了 Python 就不认（__init__ 改叫 initialize，
#              建对象时就没人调用它了）；字段名则随便取，取好读的就行。
# [2026-10-12] 知识点：可变默认值（list/dict/自定义类实例）必须走
#              field(default_factory=...)，每次建对象现造一个；
#              直接写 = [] 会让所有对象共享同一个列表（a 打标签 b 也有）。
#              int/str/None 不可变，直接写默认值没事。
@dataclass
class CleanStats:
    """一次清洗各环节删掉了多少东西。用于数据报告，也用于自检。"""

    raw_lines: int = 0
    empty_lines: int = 0
    killed_lines: int = 0        # 整行删除的噪音行
    stripped_lines: int = 0      # 做了行内切除的行
    residue_lines: int = 0       # 切完只剩残渣而丢弃的行
    kept_lines: int = 0

    rule_hits: dict[str, int] = field(default_factory=dict)


@dataclass
class CleanResult:
    # stats 用 CleanStats 当工厂：可变对象都要现造，自定义类也一样。
    # book_title 是可选信息，str | None 默认 None，扫不到书名就保持 None。
    lines: list[str] = field(default_factory=list)
    stats: CleanStats = field(default_factory=CleanStats)
    book_title: str | None = None


@dataclass
class Chapter:
    title: str
    body: str
    book: str = ""

    # [2026-10-12] 知识点：@property 让方法用起来像属性（ch.chars 不加括号），
    #              每次访问现场算，不用存一份可能过期的冗余数据。
    # [2026-10-12] 知识点：A if 条件 else B 是三元表达式；空字符串布尔值是
    #              False，所以 if self.title 等价于"标题非空"。
    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}" if self.title else self.body

    @property
    def chars(self) -> int:
        return len(self.body)


# ============================================================
# 1. 读文件
# ============================================================
# [2026-10-12] 知识点：-> tuple[str, str] 是返回值类型注解，写给 IDE 和人看，
#              解释器只存着、不做类型检查。但注解默认【会被求值】：
#              实测没有 from __future__ import annotations 时，注解里写一个
#              不存在的类型名，会在 def 那一刻就抛 NameError。
#              本文件开头有那一句，所以注解里可以先写还没定义的名字。
def read_text(path: str | Path) -> tuple[str, str]:
    """读文件并自动识别编码，返回 (文本, 编码名)。

    策略（先看证据，再猜）：
      1. 有 BOM 就按 BOM 解——这是文件自己声明的编码。BOM 还顺便被去掉，
         否则 \ufeff 会混进正文变成词表里的怪 token。
      2. 没有 BOM 时，依次试 NO_BOM_FALLBACKS（utf-8 严格，然后 gbk）。
      3. 全都不行就报错，绝不静默降级（errors="replace" 会掩盖问题）。

    为什么不能只用「依次 try/except」：Python 的 utf-16 在没有 BOM 时
    不会报错，而是按小端硬解。短文件万一"解成功"就是静默乱码，
    查都查不出来，所以 utf-16 必须在试错之前先靠 BOM 认出来。
    """
    # 先按字节读（类似 fread 进 char[]），此时还没碰编码；
    # 直接 open().read() 会当场解码，编码不对连补救机会都没有。
    raw = Path(path).read_bytes()

    # 1) BOM 优先。utf-8-sig 的 -sig 后缀会顺手把 BOM 吃掉。
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16"), "utf-16"

    # 2) 无 BOM：按实测过的顺序试，解失败抛 UnicodeError 就换下一个。
    #    注意 utf-16 故意【不在】这个列表里：没有 BOM 时它不会报错，
    #    而是按小端硬解出乱码 —— 所以 utf-16 只能靠 BOM 判断。
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
    # [2026-10-12] 知识点：re.sub(pat, "", line) 把匹配处替换成空（挖掉），
    #              字符串不可变，它返回新字符串，所以要接回来再赋回 line。
    # [2026-10-12] 知识点：dict.get(键, 0) 在键不存在时返回 0 而不是抛
    #              KeyError，计数套路：d[k] = d.get(k, 0) + 1。
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

    # 挖掉 URL 后常留下连续空格（"详情请访问 www.x.com 获取" → "详情请访问  获取"），
    # 合并成一个，别让它变成两个 token。\u3000 是全角空格。
    line = re.sub(r"[ \t\u3000]{2,}", " ", line)
    return line.strip()


def is_noise_line(line: str) -> bool:
    """整行是不是纯噪音（没有任何正文价值）。"""
    # [2026-10-12] 知识点：any(...) 只要一个为真就返回 True，命中即短路；
    #              search 在全行任意位置找，match 只从行首试。
    #              匹配失败返回 None，布尔值是 False，正好直接当条件用。
    return any(re.search(pat, line) for pat in LINE_KILL_PATTERNS)


def clean_lines(text: str) -> CleanResult:
    """三层清洗：整行删除 → 行内切除 → 残渣丢弃。

    返回 CleanResult（干净的行 + 统计 + 书名）。
    """
    result = CleanResult()
    stats = result.stats  # 起个短别名，指向同一个对象，不是拷贝

    # splitlines() 按换行切成行的列表，\r\n / \r 各种风格都认
    raw_lines = text.splitlines()
    stats.raw_lines = len(raw_lines)

    for raw in raw_lines:
        # 先 strip：后面正则多带 ^ 锚点，行首有空格会匹配失败
        line = raw.strip()

        # --- 空行：真实语料里 66% 的行是空行（原文"一行一段"）---
        if not line:
            stats.empty_lines += 1
            continue

        # --- 书名行：抽出来当书名，不进正文 ---
        # 三个条件同时满足才算：整行是《X》、还没记过书名、正文还是空的。
        # 后两条防止把正文中间提到的别的书名误当成本书书名。
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

        # --- 第三层：只有「原来挺长、切完只剩渣」才丢 ---
        # 用手术前长度做准入，不误伤本来就短的正文（"轰！"）。
        if before_len >= KEEP_IF_STRIPPED_FROM and after_len < MIN_KEPT_AFTER_STRIP:
            stats.residue_lines += 1
            continue

        # 空掉的行丢弃；剩下的不管多短都保留（"轰！"是正文）
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
    # [2026-10-12] 知识点：match 成功返回 Match 对象、失败返回 None。
    #              函数声明了 -> bool，所以用 bool(...) 显式转一次：
    #              调用方拿到的就一定真是 True/False，而不是 None
    #              （None 在 if 里也算假，但传给别人时会和别的"没值"混在一起）。
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

    # [2026-10-12] 知识点：切片 line[:cut] 取前 cut 个字符（下标 0 到 cut-1）。
    #              切片越界【不报错】，只会静默少取或多取空 —— 这是 Python 最常见的坑之一，
    #              所以下面用 cut = -1 当前哨兵，靠它判断"没找到切点"。
    # [2026-10-12] 知识点：enumerate(可迭代, start=1) 一次给出下标和元素；
    #              start=1 让下标从 1 开始，正好当切片位置用。
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
    - 正文为空的章节不产出（避免"只有标题没内容"的脏章节污染数据）。
    """
    # [2026-10-12] 知识点：函数体里只要出现 yield，它就变成【生成器函数】。
    #              调用它不会立刻执行任何代码，而是马上返回一个生成器对象；
    #              之后每被 next() 一次，才跑到下一个 yield 并"暂停"在那里。
    #              好处：1442 章能一章一章往外吐，不必一次性全塞进内存。
    #              代价：生成器只能从头到尾遍历一次，想再遍历得重新调用函数。
    # [2026-10-12] 知识点：title or "" —— or 在左边为假时返回右边。
    #              None 和空字符串都是假，所以这里等价于 title if title else ""。
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
    # [2026-10-12] 知识点：函数名前面加一个下划线，是约定"这是内部函数，
    #              外部别直接调"（Python 不强制，纯默契）。
    # [2026-10-12] 知识点：\s 是空白，\W 是"非单词字符"（标点等），+ 表示至少一个。
    #              注意 \w 本来就包含下划线，所以 \W 是【不】匹配下划线的 ——
    #              这也是为什么这里要额外把 _ 写进字符类里，否则 a_b 不会被归一化掉。
    return re.sub(r"[\s\W_]+", "", s)


def dedup_paragraphs(
    paragraphs: Iterable[str],
    min_len: int = DEDUP_MIN_LEN,
    seen: set[str] | None = None,
) -> tuple[list[str], int]:
    """段落级精确去重。

    只对长度 >= min_len 的段落去重：短段落（拟声词、口头禅）是文风。
    返回 (保留的段落, 删掉的段落数)。

    seen 必须由调用方持有并跨书传进来——否则每本书都新建一个集合，
    跨书去重就失效了（容易踩的坑：函数内部自己 set() 看起来很干净，
    实际把「全局去重」偷偷降级成了「单本去重」）。
    """
    # [2026-10-12] 知识点：set 用哈希表，in 查找近似 O(1)；list 是逐个比对，O(n)。
    #              实测 20 万条数据查 2000 次：set 0.22 ms，list 999 ms，差约 4500 倍。
    #              去重时每个段落都要查一次"见过没"，所以这里必须用 set。
    # [2026-10-12] 知识点：if seen is None: seen = set() 是"哨兵值"写法 ——
    #              默认参数不能写 = set()（那样所有调用会共享同一个集合），
    #              所以用 None 当哨兵，进函数再现造一个。
    # [2026-10-12] 知识点：hashlib.md5 只吃字节，所以先 .encode("utf-8")；
    #              hexdigest() 把摘要转成 32 个十六进制字符，比存字节更好比对和打印。
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
    """章节级去重：正文归一化后完全一致的精确重复章节。

    用归一化后的正文做指纹，能吃掉「只差空白/标点/编码」的重复章节。

    已知边界（有意不做的）：如果两个源在正文里插了不同的水印
    （甲源是「正文」，乙源是「正文 16k小说」），归一化后仍不同，
    这个函数抓不到。要抓这种需要 SimHash / MinHash 近似去重，
    代价是每章存一份 shingle 指纹、且 Python 里算得慢。
    单本书语料里没有这种场景（实测重复率上限 0.43%），
    所以等 v1 扩到多书语料时再实现，不提前优化。
    """
    # [2026-10-12] 知识点：这里比对的是 _norm 归一化后的正文（去掉了空白和标点），
    #              所以"只差排版/标点"的重复章节也能被抓出来。
    #              前面的 len(norm) < 50 是护栏：太短的章节直接放行，
    #              避免把「楔子」这种本来就短的章节误判成重复删掉。
    if seen is None:
        seen = set()
    kept: list[Chapter] = []
    removed = 0

    for ch in chapters:
        norm = _norm(ch.body)
        if len(norm) < 50:            # 太短的章节不去重，避免误杀
            kept.append(ch)
            continue
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        kept.append(ch)

    return kept, removed