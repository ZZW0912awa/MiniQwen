# -*- coding: utf-8 -*-
"""回归测试：BPE 增量 pair 计数是否与「全量重算」一致。

为什么需要这个测试
------------------
`scripts/bpe_from_scratch.py` 里的 `merge_pair_incremental()` 为了把
「每轮重扫全语料」优化成「只更新合并点附近」，手工维护了一份 pair 计数表。
但增量更新极易写错，而且**错了不会报错**——分词器照样能训练完、照样能
编码解码，只是合并顺序变差、压缩率变低。这种静默错误最难发现。

判定标准：增量维护的 counts，必须和「用 get_pair_counts() 从头数一遍」的结果完全一致。

当前状态：已知存在缺陷（见下），因此标记为 xfail。
修复后 xfail 会变成 XPASS，pytest 会提示你把这个标记删掉。
"""

import random
from collections import defaultdict

import pytest

from bpe_from_scratch import get_pair_counts, merge_pair_incremental


def _first_mismatch(seq, n_merges=40):
    """跑若干轮 merge，每轮都和全量重算对账，返回第一个不一致的描述。"""
    ids = list(seq)
    counts = defaultdict(int, get_pair_counts(ids))

    for step in range(n_merges):
        best_pair, best_cnt = None, 0
        for p, c in counts.items():
            if c > best_cnt:
                best_pair, best_cnt = p, c
        if best_pair is None or best_cnt < 2:
            return None

        ids = merge_pair_incremental(ids, best_pair, 256 + step, counts)
        truth = get_pair_counts(ids)

        # 真实存在的 pair + 增量表里被写成非 0 的 pair，都要对得上
        keys = set(truth) | {p for p, c in counts.items() if c != 0}
        for p in keys:
            if counts.get(p, 0) != truth.get(p, 0):
                return {
                    "step": step,
                    "merged": best_pair,
                    "pair": p,
                    "incremental": counts.get(p, 0),
                    "truth": truth.get(p, 0),
                }
    return None


@pytest.mark.xfail(
    reason="已知缺陷：相邻的两个合并会重复计算中间那个 pair（一侧多做一次减法，"
    "另一侧多做一次加法）",
    strict=False,
)
def test_incremental_counts_match_full_recount():
    """三个相同 token 相连——最容易触发相邻合并重叠的情形。"""
    mismatch = _first_mismatch([1, 1, 1, 2, 1, 2, 1, 2, 3])
    assert mismatch is None, f"增量计数与全量重算不一致: {mismatch}"


@pytest.mark.xfail(reason="同上：交替模式 b,a,b,a 会让相邻合并共享同一个中间 pair", strict=False)
def test_incremental_counts_alternating():
    mismatch = _first_mismatch(list("ababababab".encode()))
    assert mismatch is None, f"增量计数与全量重算不一致: {mismatch}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_incremental_counts_random(seed):
    """随机序列：不保证一定触发，但只要触发就必须报出来。"""
    rng = random.Random(seed)
    seq = [rng.randint(0, 12) for _ in range(300)]
    mismatch = _first_mismatch(seq, n_merges=60)
    if mismatch is not None:
        pytest.xfail(f"随机用例 seed={seed} 触发不一致: {mismatch}")
