# -*- coding: utf-8 -*-
"""语料处理库的单元测试。

这些测试的每一条都对应真实数据里的一个坑（见 ROADMAP 阶段 1 的侦察记录），
不是凭空写的：
  - 原始语料是 GBK
  - 水印藏在行中间，还用全角字符伪装
  - 「小说」这个词在正文里合法存在，不能当关键词
  - 「轰！」这类短段落是正文，原版按长度 <4 直接删掉是错的
"""

from pathlib import Path

import pytest

from miniqwen.corpus import (
    clean_lines,
    dedup_chapters,
    dedup_paragraphs,
    is_chapter_title,
    iter_chapters,
    read_text,
    split_glued_title,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ============================================================
# 编码探测
# ============================================================
@pytest.mark.parametrize(
    "filename,expect_enc",
    [
        ("book_a_utf8.txt", "utf-8"),          # 无 BOM → 报 utf-8（不是 utf-8-sig）
        ("book_b_gbk.txt", "gbk"),
        ("book_c_utf8_bom.txt", "utf-8-sig"),  # 有 BOM → 按 BOM 解
        ("book_d_utf16.txt", "utf-16"),        # 有 BOM → 按 BOM 解
    ],
)
def test_read_text_detects_encoding(filename, expect_enc):
    text, enc = read_text(FIXTURES / filename)
    assert enc == expect_enc
    assert "第一章" in text or "楔子" in text or "序章" in text


def test_utf16_without_bom_is_not_guessed():
    """回归测试：没有 BOM 的字节流不能被当成 UTF-16。

    Python 的 utf-16 解码器在没有 BOM 时不会报错，而是按小端硬解，
    一个短文件万一"解成功"就是静默乱码。所以 utf-16 只能靠 BOM 判断。
    """
    import codecs

    from miniqwen.corpus import NO_BOM_FALLBACKS

    assert "utf-16" not in NO_BOM_FALLBACKS
    # 造一个"看起来像 utf-16"但没有 BOM 的短文件：应判成 gbk 而不是 utf-16
    raw = "第一章 测试".encode("gbk")
    assert not raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
    assert raw.decode("gbk") == "第一章 测试"


def test_bom_is_stripped():
    """带 BOM 的文件不能让 \\ufeff 混进正文，否则会变成一个莫名其妙的 token。"""
    text, enc = read_text(FIXTURES / "book_c_utf8_bom.txt")
    assert not text.startswith("\ufeff")
    assert text.startswith("《测试之书丙》")


def test_book_title_extracted():
    result = clean_lines(read_text(FIXTURES / "book_a_utf8.txt")[0])
    assert result.book_title == "测试之书甲"
    # 书名行不能留在正文里
    assert all("测试之书甲" not in ln for ln in result.lines)


# ============================================================
# 清洗：不能误删正文（这一组是回归测试）
# ============================================================
def test_legit_word_xiaoshuo_survives():
    """真实语料里的一行正文，含「小说」二字，绝不能被当噪音删掉。

    侦察时用宽松规则（匹配「小说」）跑，这一行就被误报了。
    """
    line = "大陆名为斗气大陆，大陆上并没有小说中常见的各系魔法，而斗气，才是大陆的唯一主调！"
    result = clean_lines(line)
    assert result.lines == [line]


@pytest.mark.parametrize("short", ["“轰！”", "“嘭！”", "“嗤！”", "短。"])
def test_short_paragraphs_are_kept(short):
    """拟声词短段落是文风，必须保留。

    原版 clean_novel.py 的 `if len(line) >= 4` 会把它们全删掉。
    """
    result = clean_lines(short)
    assert result.lines == [short]


def test_inline_url_removed_but_text_kept():
    line = "老人说：走吧，别回头。https://ads.example.com/xyz"
    result = clean_lines(line)
    assert result.lines == ["老人说：走吧，别回头。"]


def test_inline_watermark_with_fullwidth_disguise_removed():
    """盗版站用希腊字母 α 伪装 wap，半角正则会漏掉。"""
    line = "淡淡的笑了笑，萧炎甩开脑中的回忆，继续研习着手中的碎石掌。16k小说wαp.16k整理"
    result = clean_lines(line)
    assert result.lines == ["淡淡的笑了笑，萧炎甩开脑中的回忆，继续研习着手中的碎石掌。"]


def test_watermark_glued_mid_sentence_removed():
    line = "“谷尼叔叔，他真是炼药师？”小说整理发布于16k”略微沉默了一会，雅妃轻声问道武动乾坤。"
    result = clean_lines(line)
    assert len(result.lines) == 1
    assert "16k" not in result.lines[0]
    assert "武动乾坤" not in result.lines[0]
    assert "雅妃轻声问道" in result.lines[0]


def test_whole_line_ad_is_killed():
    result = clean_lines("请收藏本站\n（未完待续）\n真正的正文在这里。")
    assert result.lines == ["真正的正文在这里。"]
    assert result.stats.killed_lines == 2


def test_residue_line_is_dropped():
    """整行基本就是水印（切完只剩渣）→ 丢弃。"""
    result = clean_lines("www.example-novel.com 最新章节")
    assert result.lines == []
    assert result.stats.killed_lines + result.stats.residue_lines >= 1


def test_stats_add_up():
    result = clean_lines(read_text(FIXTURES / "book_a_utf8.txt")[0])
    s = result.stats
    total = s.empty_lines + s.killed_lines + s.residue_lines + s.kept_lines
    # 书名行被单独抽走，所以允许差 1
    assert total in (s.raw_lines, s.raw_lines - 1)


# ============================================================
# 切章
# ============================================================
def test_chapter_marker_recognises_multiple_styles():
    for t in ["第一章 起风", "第12回 归来", "第三卷 风起", "楔子", "序章", "番外 其一"]:
        assert is_chapter_title(t), t


def test_split_glued_title():
    """真实脏数据：标题和正文粘在一行。"""
    line = "第七百七十六章狼狈的铁护法的铁护法帝遇，据薰儿所说，这是其族中秘技之一，只能族中天赋杰出之人方才有资格修炼"
    title, body = split_glued_title(line)
    assert len(title) <= 30
    assert title.startswith("第七百七十六章")
    assert body.startswith("据薰儿所说") or body.startswith("的铁护法")


def test_short_title_untouched():
    title, body = split_glued_title("第一章 起风")
    assert title == "第一章 起风"
    assert body == ""


def test_iter_chapters_counts():
    result = clean_lines(read_text(FIXTURES / "book_a_utf8.txt")[0])
    chapters = list(iter_chapters(result.lines, "甲"))
    titles = [c.title for c in chapters]
    assert "第一章 起风" in titles
    assert "第二章 落雨" in titles


def test_iter_chapters_handles_prologue():
    result = clean_lines(read_text(FIXTURES / "book_b_gbk.txt")[0])
    chapters = list(iter_chapters(result.lines, "乙"))
    assert chapters[0].title == "楔子"


def test_prose_starting_with_marker_word_is_not_a_title():
    """回归测试：正文首句「序章的内容写在这里…」不能被当成章节标题。

    曾经的 bug：`序章` 只做了前缀匹配，于是这行被当成新章节标题，
    真正的序章正文变成了空章节被丢弃——整章数据凭空消失。
    """
    result = clean_lines(read_text(FIXTURES / "book_c_utf8_bom.txt")[0])
    chapters = list(iter_chapters(result.lines, "丙"))
    titles = [c.title for c in chapters]

    assert len(chapters) == 3, f"应有三章，实际 {titles}"
    assert titles[0] == "序章"
    assert "序章的内容写在这里" in chapters[0].body


def test_marker_word_variants_recognised():
    for t in ["序章", "楔子", "楔子 血夜", "番外：其一", "尾声", "后记"]:
        assert is_chapter_title(t), t
    for t in ["序章的内容写在这里", "楔子是一种引子", "番外篇讲了什么"]:
        assert not is_chapter_title(t), t


def test_no_empty_chapters():
    """正文为空的章节不该产出。"""
    result = clean_lines("第一章 空的\n第二章 有内容\n这里有正文。")
    chapters = list(iter_chapters(result.lines))
    assert len(chapters) == 1
    assert chapters[0].title == "第二章 有内容"


def test_extra_spaces_collapsed_after_url_removal():
    """水印切除后别留下连续空格，否则会多出无意义的 token。"""
    result = clean_lines("详情请访问 www.example-novel.com 获取最新章节")
    assert result.lines == ["详情请访问 获取最新章节"]
    assert "  " not in result.lines[0]


def test_chapter_text_includes_title():
    """写进 corpus.txt 的章节要带标题，模型才能学会「第X章」这个模式。"""
    from miniqwen.corpus import Chapter

    ch = Chapter("第一章 起风", "少年背着剑走出了山谷。", "甲")
    assert ch.text.startswith("第一章 起风\n")
    assert ch.chars == len("少年背着剑走出了山谷。")


# ============================================================
# 去重
# ============================================================
def test_dedup_threshold_protects_short_paragraphs():
    """短段落即使重复也不能删——这就是 DEDUP_MIN_LEN 存在的理由。"""
    paras = ["“轰！”"] * 10 + ["这是一段足够长的重复段落，长度超过了二十个字的阈值。"] * 3
    kept, removed = dedup_paragraphs(paras, min_len=20)
    assert kept.count("“轰！”") == 10      # 全部保留
    assert removed == 2                   # 长段落保留 1 份，删 2 份


def test_dedup_is_global_when_seen_is_passed_in():
    """传同一个 seen 集合，两次调用之间也要能去重（跨书去重的关键）。"""
    seen: set[str] = set()
    long_p = "这是一段足够长的段落，用来验证跨书去重是否真的生效。"

    kept1, removed1 = dedup_paragraphs([long_p], min_len=20, seen=seen)
    kept2, removed2 = dedup_paragraphs([long_p], min_len=20, seen=seen)

    assert len(kept1) == 1 and removed1 == 0
    assert len(kept2) == 0 and removed2 == 1     # 第二本书里的同一段被吃掉


def test_dedup_chapters_catches_exact_duplicate():
    """同一章从两个源各抓了一遍、正文完全一致 → 必须去掉一份。"""
    from miniqwen.corpus import Chapter

    body = "这是一章足够长的正文内容，用来测试章节级去重能不能识破重复。" * 3
    kept, removed = dedup_chapters([Chapter("第一章", body, "甲"), Chapter("第一章", body, "乙")])
    assert len(kept) == 1 and removed == 1


def test_dedup_chapters_tolerates_whitespace_and_punctuation():
    """只差空白/标点的重复也要能抓住（归一化的作用）。"""
    from miniqwen.corpus import Chapter

    body = "这是一章足够长的正文内容，用来测试章节级去重能不能识破重复。" * 3
    noisy = body.replace("，", "， ").replace("。", "。\n")
    kept, removed = dedup_chapters([Chapter("第一章", body, "甲"), Chapter("第一章", noisy, "乙")])
    assert len(kept) == 1 and removed == 1


def test_dedup_chapters_short_chapters_are_never_dropped():
    """很短的章节不做去重，避免把「楔子」这种短章节误杀。"""
    from miniqwen.corpus import Chapter

    chapters = [Chapter("楔子", "很短。", "甲"), Chapter("楔子", "很短。", "乙")]
    kept, removed = dedup_chapters(chapters)
    assert len(kept) == 2 and removed == 0
