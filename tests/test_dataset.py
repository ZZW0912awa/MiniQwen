# -*- coding: utf-8 -*-
"""二进制语料 + 加载器的测试。

重点验三件事：
  1. **y 必须是 x 右移一位** —— off-by-one 不会报错，只会让模型永远学错
  2. **dtype 必须是 int64** —— uint16 喂不进 nn.Embedding
  3. **越界要显式报错** —— memmap 切片越界会静默返回短数组，
     而 np.stack 遇到长度不齐才报一个莫名其妙的错
"""

import numpy as np
import pytest

from miniqwen.dataset import DEFAULT_DATA_DIR, BinnedDataset, TokenFile

pytestmark = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "meta.json").exists(),
    reason="还没生成二进制语料（先跑 py scripts/prepare_data.py）",
)

BLOCK = 64
BATCH = 4


@pytest.fixture(scope="module")
def ds():
    return BinnedDataset()


# ============================================================
# 基本信息
# ============================================================
def test_meta_and_sizes(ds):
    assert len(ds.train) == ds.meta["train"]["tokens"]
    assert len(ds.val) == ds.meta["val"]["tokens"]
    assert ds.meta["dtype"] == "uint16"
    assert ds.meta["bytes_per_token"] == 2


def test_file_size_matches_token_count(ds):
    """文件大小必须正好是 token 数 × 2 —— 多一个字节都说明写错了。"""
    p = ds.data_dir / ds.meta["train"]["file"]
    assert p.stat().st_size == ds.meta["train"]["tokens"] * 2


def test_vocab_size_fits_in_dtype(ds):
    """所有 id 必须小于词表大小，且装得进 uint16。"""
    for split in ("train", "val"):
        d = ds.get_file(split)
        assert int(d.data.max()) < ds.vocab_size, f"{split} 里有超出词表的 id"
        assert int(d.data.max()) <= 65535


def test_no_negative_ids(ds):
    """uint16 是无符号的 —— 若写文件时有负值，读回来会变成 6 万多。"""
    for split in ("train", "val"):
        assert int(ds.get_file(split).data.min()) >= 0


# ============================================================
# get_batch —— 核心
# ============================================================
@pytest.mark.parametrize("split", ["train", "val"])
def test_batch_shapes(ds, split):
    x, y = ds.get_batch(split, BLOCK, BATCH)
    assert x.shape == (BATCH, BLOCK)
    assert y.shape == (BATCH, BLOCK)


def test_batch_dtype_is_int64(ds):
    """nn.Embedding 要求索引是 int64；uint16 直接喂会报错。"""
    x, y = ds.get_batch("train", BLOCK, BATCH)
    assert x.dtype == np.int64
    assert y.dtype == np.int64


def test_y_is_x_shifted_by_one(ds):
    """核心不变式：把 y 的最后一列去掉，必须等于 x 去掉第一列。

    这个错了不会报错 —— 形状对、能训练、loss 也降，但模型学的是错位的东西。
    """
    rng = np.random.default_rng(0)
    x, y = ds.get_batch("train", BLOCK, BATCH, rng=rng)
    np.testing.assert_array_equal(y[:, :-1], x[:, 1:])


def test_batch_values_are_valid_ids(ds):
    x, y = ds.get_batch("train", BLOCK, BATCH)
    assert x.min() >= 0 and x.max() < ds.vocab_size
    assert y.min() >= 0 and y.max() < ds.vocab_size


# ============================================================
# 边界与报错
# ============================================================
def test_invalid_split_raises(ds):
    with pytest.raises(ValueError, match="train"):
        ds.get_batch("valid", BLOCK, BATCH)      # 拼错了，不能静默当 val 用


def test_window_out_of_range_raises(ds):
    with pytest.raises(IndexError):
        ds.train.window(len(ds.train) - 10, 100)


def test_block_size_too_large_raises():
    """val 只有 5.8 万 token；让它开一个比整个文件还长的窗口必须报错。"""
    ds = BinnedDataset()
    with pytest.raises(ValueError, match="装不下"):
        ds.get_batch("val", len(ds.val) + 10, 2)


def test_missing_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        TokenFile(tmp_path / "nope.bin")
    assert "prepare_data" in str(e.value), "报错要告诉用户怎么修"


# ============================================================
# memmap 语义
# ============================================================
def test_window_is_a_view_not_a_copy(ds):
    """window() 返回 memmap 视图（不复制）—— 这是"内存零占用"的关键。

    如果哪天有人改成 np.array(...) 复制，这个测试会红。
    """
    w = ds.train.window(0, BLOCK)
    assert isinstance(w, np.memmap), "window 不该复制成普通数组"


def test_two_dataset_instances_share_file(ds):
    """两个实例读同一文件，数据必须一致（说明不是各自缓存了一份）。"""
    other = BinnedDataset()
    np.testing.assert_array_equal(ds.train.data[:100], other.train.data[:100])
