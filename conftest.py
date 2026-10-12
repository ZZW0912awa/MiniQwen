# -*- coding: utf-8 -*-
"""让 tests/ 能直接 import scripts/ 下的脚本（scripts 不是包）。

【为什么需要这个文件】
pytest 默认只把「测试文件自己所在的目录」(tests/) 放进 sys.path，
而 miniqwen/ 在它上一级、bpe_from_scratch.py 在 scripts/ 下，两边都找不到，
于是报 ModuleNotFoundError: No module named 'miniqwen'。
而 conftest.py 里的代码会在 pytest 收集测试【之前】执行，所以能救回来。
"""

import sys
from pathlib import Path

# ============================================================
# 一、定位「本文件自己」在哪里
# ============================================================
#   Path(__file__)   __file__ 是 Python 自动注入的变量 = 本文件的路径
#                    （相当于 C++ 的 __FILE__ 宏）；Path(...) 把它从字符串变成路径对象
#   .resolve()       转成绝对路径，并消掉 ".."、"." 和软链接
#   .parent          取上一级目录（本文件就在仓库根目录，所以 ROOT = 仓库根）
#
# —— 关于 .parent 和 .parents 的区别（已用实验验证）——
#   .parent                == .parents[0]     ← 注意！parents 是【从 0 开始】的
#   .parent.parent         == .parents[1]
#   .parent.parent.parent  == .parents[2]
#   也就是说 .parents[i] 里的 i 就是「往上跳几级」，不是跳几级+1。
#
#   两者还有一个重要差别 —— 越界时的行为：
#     Path("C:/a").parents[99]                   -> IndexError，当场报错（有边界检查）
#     Path("C:/a").parent.parent.parent...       -> 一直返回 C:\，【不报错】，静默出错
#   所以想往上跳多级时，用 .parents[i] 更安全：层级写错会当场炸，
#   而不是悄悄给你一个错误的目录 —— 后者才是最难查的 bug。
#
# 我的原始理解（2026-10-12）：
#   「可以使用 .parent.parent... 也可以使用 .parents[0...] 保证不会跳到非法目录」
#   修正：.parents[0] 就是 .parent，不是上两级；«保证不越界» 这个好处属于
#        .parents[i]（会抛 IndexError），.parent 连写反而没有边界保护。
ROOT = Path(__file__).resolve().parent

# ============================================================
# 二、把需要的目录塞进 sys.path
# ============================================================
# sys.path 是 Python 的「模块搜索路径清单」，清单里的每一项是一个【目录】
# （严格说也可以是 zip 文件 —— 标准库路径里的 python312.zip 就是这种用法，
#   Python 支持从 zip 里直接 import 模块）。
# 执行 import xxx 时，Python 就按这个清单【从头到尾】在每个目录里找
# xxx.py 或 xxx/ 目录，所以谁排在前面谁的优先级高
# —— 这就是用 insert(0, ...) 插到最前面的原因。
#
# 两个容易忽略的细节：
#   str(p)   sys.path 里存的都是【字符串】，而 p 是 Path 对象，
#            两者不相等、也不会自动转换，必须显式转
#   if 判断  防重复。这个文件可能被导入多次，不加判断会把同一目录塞进去好几遍
#
# 我的原始理解（2026-10-12）：
#   「将 ROOT 和 ROOT 下的 scripts 内不在 sys.path 的文件加入到最前面」
#   修正：加进去的是【目录】（ROOT 和 ROOT/scripts 这两个目录），
#        不是「scripts 内的文件」。sys.path 是「去哪找」的清单，不是「有哪些文件」的清单。
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
