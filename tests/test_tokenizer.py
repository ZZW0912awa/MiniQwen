# -*- coding: utf-8 -*-
"""分词器的单元测试。

重点验三件事：
  1. **无损**——字节级 BPE 对任何 Unicode 文本都必须能原样还原，
     包括中文、英文、emoji、生僻字、控制字符。这一条挂了，语料就是脏的。
  2. **特殊 token 是单个 token**——不能被拆成 < | i m ... 好几个。
  3. **对话模板的 loss mask 正确**——提示词部分必须是 -100。
"""

import pytest

from miniqwen.tokenizer import (
    DEFAULT_TOKENIZER_PATH,
    IM_END,
    IM_START,
    SPECIAL_TOKENS,
    NovelTokenizer,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_TOKENIZER_PATH.exists(),
    reason="还没训练分词器（先跑 py scripts/train_tokenizer.py）",
)


@pytest.fixture(scope="module")
def tok():
    return NovelTokenizer()


# ============================================================
# 基本信息
# ============================================================
def test_vocab_size_fits_uint16(tok):
    """词表必须装得进 uint16 —— 阶段 3 要把 token 存成 uint16 的 .bin。

    这是硬约束，不是魔数：UINT16_MAX = 65535，超了就得换 uint32，
    数据体积直接翻倍。写死 "== 16384" 反而脆弱：换 --vocab-size 重训就红。
    """
    assert 1024 <= tok.vocab_size <= 65535, (
        f"词表 {tok.vocab_size} 超出 uint16 范围，阶段 3 的 .bin 会装不下"
    )


def test_special_tokens_have_distinct_ids(tok):
    ids = [tok.token_to_id(s) for s in SPECIAL_TOKENS]
    assert all(i is not None for i in ids), "有特殊 token 没注册进词表"
    assert len(set(ids)) == len(SPECIAL_TOKENS), "特殊 token 的 id 撞车了"


def test_special_tokens_are_not_split(tok):
    """<|im_start|> 必须编成 1 个 token，不能变成 < | i m _ s ... 一串。"""
    for s in SPECIAL_TOKENS:
        ids = tok.encode(s)
        assert len(ids) == 1, f"{s} 被拆成了 {len(ids)} 个 token"
        # decode 默认 skip_special_tokens=True，会把特殊 token 抹掉（返回空串），
        # 所以必须显式要求保留，才能验证编进去的确实是原字符串
        assert tok.decode(ids, skip_special_tokens=False) == s


# ============================================================
# 无损性 —— 这一组是最重要的
# ============================================================
@pytest.mark.parametrize(
    "text",
    [
        "萧炎点了点头，淡淡的道：“走吧。”",
        "第一章 陨落的天才\n“斗之力，三段！”",
        "Hello, world! 这是中英混排 mixed text 12345.",
        "emoji 也能编吗？🤔🔥🐳 试试看",
        "生僻字：齉龘靐齉爩 还有 CJK 扩展区 𠀀𠀁",
        "标点全来一遍：，。！？；：、“”‘’（）《》【】——……",
        "",                      # 空串
        " ",                     # 只有空格
        "\n\n\n",                # 只有换行
        "a",                     # 单字符
    ],
)
def test_roundtrip_lossless(tok, text):
    """编码再解码，必须一字不差。"""
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_on_corpus_sample(tok):
    """拿一段真实语料做往返，长度够长才算数。"""
    text = DEFAULT_TOKENIZER_PATH.parent.parent / "processed" / "corpus.txt"
    if not text.exists():
        pytest.skip("没有 corpus.txt")
    sample = text.read_text(encoding="utf-8")[500_000:502_000]
    assert tok.decode(tok.encode(sample)) == sample


# ============================================================
# 压缩率
# ============================================================
def test_compression_beats_byte_level(tok):
    """中文的字节级 BPE 若压缩不到 3 字节/token，说明词表没学到东西。

    参考：纯字节（不合并）时，一个汉字 = 3 字节 = 3 个 token，
    即 1 字节/token。我们实测 5.35 字节/token。
    """
    text = "萧炎缓缓抬起头来，露出一张有些清秀的稚嫩脸庞。" * 20
    n_bytes = len(text.encode("utf-8"))
    n_tok = len(tok.encode(text))
    assert n_bytes / n_tok > 3.0, f"压缩率只有 {n_bytes/n_tok:.2f} 字节/token"


def test_common_words_become_single_tokens(tok):
    """高频词应该被合并成少量 token —— 这是 BPE 学到东西的直接证据。"""
    ids = tok.encode("萧炎")
    assert len(ids) <= 2, f"「萧炎」用了 {len(ids)} 个 token，太多了"


# ============================================================
# 对话模板
# ============================================================
def test_chat_template_format(tok):
    text = tok.apply_chat_template("写一段斗气突破", "萧炎盘膝而坐。")
    assert text.startswith(f"{IM_START}user\n")
    assert f"{IM_END}\n{IM_START}assistant\n" in text
    assert text.rstrip().endswith(IM_END)
    assert "写一段斗气突破" in text and "萧炎盘膝而坐。" in text


def test_chat_template_without_response(tok):
    """只给提示词时应该停在 assistant 开头 —— 推理时就是这么喂的。"""
    text = tok.apply_chat_template("写一段斗气突破")
    assert text.endswith(f"{IM_START}assistant\n")
    assert IM_END not in text.split("assistant")[-1]


def test_encode_chat_masks_prompt(tok):
    """提示词部分的 label 必须是 -100，模型才只学「怎么答」不学「怎么背题」。"""
    ids, labels = tok.encode_chat("写一段斗气突破", "萧炎盘膝而坐。")

    assert len(ids) == len(labels)
    n_masked = sum(1 for x in labels if x == -100)
    assert n_masked > 0, "提示词完全没被 mask"
    assert n_masked < len(labels), "连回答也被 mask 了"

    # mask 的是前缀：前 n_masked 个都是 -100，之后都不是
    assert all(x == -100 for x in labels[:n_masked])
    assert all(x != -100 for x in labels[n_masked:])
    # 没被 mask 的部分必须和 input_ids 对齐
    assert labels[n_masked:] == ids[n_masked:]


# ============================================================
# 边界
# ============================================================
def test_file_not_found_gives_clear_error():
    with pytest.raises(FileNotFoundError) as e:
        NovelTokenizer("不存在的路径/tokenizer.json")
    assert "train_tokenizer" in str(e.value), "报错信息应该告诉用户怎么解决"
