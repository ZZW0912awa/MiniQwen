# -*- coding: utf-8 -*-
"""二进制语料加载器：用 np.memmap 按需读取，不把数据读进内存。

为什么不用"读成 numpy 数组"就完事
--------------------------------
`np.fromfile()` 会把整个文件读进内存。现在只有 5 MB 无所谓，
但扩到 20 本书（几千万 token）就是几百 MB，再往后就是几个 GB。
`np.memmap` 只建立**虚拟内存映射**：取哪一段，操作系统才把那几页读进来。
所以常驻内存与语料总量基本无关。

用法:
    from miniqwen.dataset import BinnedDataset

    ds = BinnedDataset()                      # 读 data/processed/
    x, y = ds.get_batch("train", block_size=512, batch_size=8)
    # x, y 都是 np.int64 的 (8, 512) 数组，直接 torch.from_numpy 就能送模型
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class TokenFile:
    """一个 uint16 的 token 文件，惰性映射。"""

    def __init__(self, path: str | Path, dtype=np.uint16):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"找不到 {self.path}\n先跑：py scripts/prepare_data.py"
            )
        self.dtype = dtype
        # mode="r" 只读；不指定 shape，一维数组，元素个数由文件大小自动推出
        self.data = np.memmap(self.path, dtype=dtype, mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"TokenFile({self.path.name}, {len(self):,} tokens)"

    def window(self, start: int, length: int) -> np.ndarray:
        """取 [start, start+length) 这段。

        注意：返回的是 **memmap 的视图**，没有复制数据。
        调用方如果要跨步长比对（比如 x 和 y），需要留意它们共享底层存储。
        """
        if start < 0 or start + length > len(self):
            raise IndexError(f"窗口越界: start={start}, length={length}, 总长={len(self)}")
        return self.data[start:start + length]


class BinnedDataset:
    """训练/验证两套 .bin 的封装，附带 meta.json 里的元信息。"""

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"找不到 {meta_path}\n先跑：py scripts/prepare_data.py"
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.train = TokenFile(self.data_dir / self.meta["train"]["file"])
        self.val = TokenFile(self.data_dir / self.meta["val"]["file"])

    def __repr__(self) -> str:
        return (f"BinnedDataset(train={len(self.train):,}, "
                f"val={len(self.val):,}, vocab={self.meta['vocab_size']:,})")

    @property
    def vocab_size(self) -> int:
        return self.meta["vocab_size"]

    def get_file(self, split: str) -> TokenFile:
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        raise ValueError(f"split 只能是 'train' 或 'val'，收到 {split!r}")

    def get_batch(
        self,
        split: str,
        block_size: int,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """随机取一个 batch。

        返回:
            x: np.int64 的 (batch_size, block_size)
            y: np.int64 的 (batch_size, block_size)，且 y 是 x 右移一位

        为什么转成 int64：uint16 的 ndarray 不能直接喂 nn.Embedding
        （它要求索引是 torch.long，也就是 int64）。转换只发生在这一小段窗口上，
        不会把整个文件载入内存。
        """
        if rng is None:
            rng = np.random.default_rng()
        d = self.get_file(split)
        n = len(d)

        # 起点上界要留够 block_size + 1：y 比 x 多一个位置
        high = n - block_size - 1
        if high <= 0:
            raise ValueError(
                f"{split} 只有 {n} 个 token，装不下一个 block_size={block_size} 的窗口"
            )
        starts = rng.integers(0, high, size=batch_size)

        x = np.stack([d.window(int(s), block_size) for s in starts])
        y = np.stack([d.window(int(s) + 1, block_size) for s in starts])
        # astype 会复制成独立的 int64 数组，避免 x/y 共享 memmap 视图带来的意外
        return x.astype(np.int64), y.astype(np.int64)
