# -*- coding: utf-8 -*-
"""生成测试样例语料。

为什么要有这个文件
------------------
真实语料只有《斗破苍穹》一本，而且它已经是「干净」的：
  - 它是 GBK，但你的代码只验证过这一种编码
  - 它没有楔子/序章/卷一 这类别的章节写法
  - 它的水印正好能被你的规则命中
换句话说，用真实语料测试，测不出批处理逻辑和编码兼容性。

所以这里手工造 3 本小书，故意把坑全塞进去：
  甲：UTF-8，标准「第X章」，含行内 URL 广告、含重复段落（长短不一）
  乙：GBK，用「楔子」开头，含水印紧贴在正文后面
  丙：UTF-8 带 BOM，用「序章」，含「（未完待续）」和一行「标题粘正文」的超长标题行

用法:
    py tests/fixtures/make_fixtures.py
"""

import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent

# ---------------- 甲：UTF-8，标准第X章，含重复段落 ----------------
BOOK_A = """《测试之书甲》

第一章 起风

少年背着剑走出了山谷。

他回头看了一眼。

他回头看了一眼。

详情请访问 www.example-novel.com 获取最新章节

第二章 落雨

雨下了三天三夜。

雨下了三天三夜，山谷里的路都成了泥。

老人说：走吧，别回头。https://ads.example.com/xyz
"""

# ---------------- 乙：GBK，楔子开头，水印紧贴正文 ----------------
# 注意：GBK 编不出希腊字母 α，所以这里只用半角 wap，别模仿甲里的全角伪装
BOOK_B = """《测试之书乙》

楔子

很久以前，有一个人住在这里。

16k小说wap.16k整理

第一章 出发

他上路了。小说整理发布于16k

第二章 归途

他回来了。
"""

# ---------------- 丙：UTF-8 带 BOM，序章，标题粘正文 ----------------
BOOK_C = """《测试之书丙》

序章

序章的内容写在这里，用来测试非「第X章」的章节写法。

第一章 归来这是一个很长的标题后面直接粘着正文，本行故意写得超过三十个字，用来测试标题与正文粘连时能不能正确切开。

第二章 再会

（未完待续）

短段。
"""

# ---------------- 丁：UTF-16（自带 BOM），测试 BOM 分支 ----------------
# Python 的 text.encode("utf-16") 会自动加 BOM，所以这条用例不需要手工拼字节。
BOOK_D = """《测试之书丁》

第一章 编码

这一本用 UTF-16 保存。

第二章 结束

编码探测必须靠 BOM 显式判断，不能靠 try/except 硬试。
"""


def write(path: Path, text: str, encoding: str, bom: bool = False) -> None:
    data = text.encode(encoding)
    if bom:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)
    print(f"  {path.name:24s} {encoding:8s} {len(data):5d} 字节")


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    print("生成测试样例:")
    write(FIXTURE_DIR / "book_a_utf8.txt", BOOK_A, "utf-8")
    write(FIXTURE_DIR / "book_b_gbk.txt", BOOK_B, "gbk")
    write(FIXTURE_DIR / "book_c_utf8_bom.txt", BOOK_C, "utf-8", bom=True)
    write(FIXTURE_DIR / "book_d_utf16.txt", BOOK_D, "utf-16")
    print("完成。注意 book_b 是 GBK，编辑器打开可能是乱码，这是故意的。")


if __name__ == "__main__":
    sys.exit(main())
