# -*- coding: utf-8 -*-
"""让 tests/ 能直接 import scripts/ 下的脚本（scripts 不是包）。

pytest 默认只把测试文件所在的 tests/ 加进 sys.path，
找不到上一级的 miniqwen/ 和 scripts/ 里的模块，会报 ModuleNotFoundError。
本文件在 pytest 收集测试【之前】执行，所以能在这里补救。
"""

import sys
from pathlib import Path

# ---- 一、定位本文件 ----
# Path(__file__)  本文件路径（类似 C++ 的 __FILE__）
# .resolve()      转绝对路径，消掉 ".." 和软链接
# .parent         上一级目录 —— 本文件就在仓库根，所以 ROOT 就是仓库根
#
# [2026-10-12] 知识点：.parents 从 0 开始索引，.parents[0] 就等于 .parent；
#              .parents[i] 里的 i 就是"往上跳几级"。
# [2026-10-12] 知识点：越界时 .parents[i] 抛 IndexError，.parent 连写却会一直
#              返回根目录且不报错 —— 静默出错比报错难查得多。
ROOT = Path(__file__).resolve().parent

# ---- 二、把需要的目录插进 sys.path 最前面 ----
# sys.path 是"去哪找模块"的清单，import 时按顺序找，排前面的优先级高
#   str(p)  必须显式转：sys.path 里存的是字符串，Path 对象和字符串不相等
#   if      防重复插入（本文件可能被导入多次）
# [2026-10-12] 知识点：sys.path 里放的是【目录】（也可以是 zip），不是"某些文件"。
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
