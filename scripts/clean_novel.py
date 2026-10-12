# -*- coding: utf-8 -*-
"""
小说数据清洗器 v4
三层清洗：整行规则 + 行内局部切除 + 高频后缀剥离
一次清洗，双格式导出：
  - .txt   纯文本（章节间空行分隔）—— 供 BPE tokenizer 训练
  - .jsonl 指令格式 {instruction, output} —— 供微调训练

用法:
    py clean_novel.py 输入文件.txt 输出名 [最低保留字数]
"""

import sys
import re
import json
from collections import Counter


def read_novel(path):
    """读取小说文件，自动尝试常见中文编码"""
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            print(f"（编码识别为 {enc}）")
            return text
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码，请手动转换后再试")


# ============ 清洗规则配置 ============

# A. 行内切除：匹配部分从句中挖掉，保留其余正文
INLINE_STRIP_PATTERNS = [
    r"https?://\S+",                          # 完整 URL
    r"www\.[A-Za-z0-9.\-]+\.(com|net|org|cn)\S*",  # www.xxx.com 形式
    r"[A-Za-z0-9.\-]+\.(com|net|org|cn)\S*",  # 裸域名 biquge.com 形式
    r"【[^】]{0,20}(?:网|小说|阅读|下载|收藏)[^】]{0,20}】",  # 【某某小说网】括号广告
    r"（[^）]{0,20}(?:网|小说|阅读|下载|收藏)[^）]{0,20}）",  # （某某小说网）
    r"[乐乐|天天|顶点|笔趣]?[小说|读书]?网?.{0,6}(?:首发|原创|独家)",  # XX网首发
]

# B. 整行删除：命中即整行是垃圾
LINE_KILL_PATTERNS = [
    r"起点中文网|笔趣阁|顶点小说|言情小说",
    r"请收藏|请记住本站|最新网址|手机阅读",
    r"本书由|更多好书|txt下载|全文阅读",
    r"更新最快|无弹窗|章节内容开始",
]


def clean_lines(text):
    """三层清洗：整行规则 → 行内切除 → 高频后缀剥离"""
    raw_lines = [line.strip() for line in text.splitlines()]

    # ---- 第一层：整行删除 ----
    stage1 = []
    for line in raw_lines:
        if not line:
            continue
        if any(re.search(p, line) for p in LINE_KILL_PATTERNS):
            continue
        stage1.append(line)

    # ---- 第二层：行内局部切除 ----
    stage2 = []
    for line in stage1:
        for p in INLINE_STRIP_PATTERNS:
            line = re.sub(p, "", line)     # 把垃圾部分挖掉，保留正文
        line = line.strip()
        # 挖完只剩残字（<4 个字符），说明整行基本是广告，丢弃
        if len(line) >= 4:
            stage2.append(line)

    # ---- 第三层：高频后缀剥离 ----
    # 对付「每章结尾粘同一句广告」：统计行尾 8 字后缀，高频的统一剥掉
    suffix_len = 5
    suffix_freq = Counter(
        line[-suffix_len:] for line in stage2 if len(line) > suffix_len * 2
    )
    bad_suffixes = {s for s, c in suffix_freq.items() if c > 10}
    stage3 = []
    stripped = 0
    for line in stage2:
        if len(line) > suffix_len * 2 and line[-suffix_len:] in bad_suffixes:
            line = line[:-suffix_len].rstrip("，。；：、 ")
            stripped += 1
        stage3.append(line)

    # ---- 清洗报告 ----
    print(f"第一层（整行删除）: 删除 {len(raw_lines) - len(stage1)} 行")
    print(f"第二层（行内切除）: 删除 {len(stage1) - len(stage2)} 行")
    print(f"第三层（后缀剥离）: 处理 {stripped} 行")
    if bad_suffixes:
        print("被剥离的高频后缀 TOP3:")
        for s in sorted(bad_suffixes, key=lambda x: -suffix_freq[x])[:3]:
            print(f"  [{suffix_freq[s]}次] ...{s}")
    return stage3


def split_chapters(lines):
    """按『第X章/回/节』标题切分，返回 [(标题, 正文), ...]"""
    chapter_re = re.compile(r"^第[零一二三四五六七八九十百千万0-9]+[章回节卷]")
    chapters, current = [], []
    for line in lines:
        if chapter_re.match(line) and current:
            chapters.append(current)
            current = []
        current.append(line)
    if current:
        chapters.append(current)

    return [(ch[0], "\n".join(ch)) for ch in chapters]


def export_txt(chapters, path, min_len):
    """导出纯文本：供 tokenizer 训练"""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for title, body in chapters:
            if len(body) < min_len:
                continue
            f.write(body + "\n\n")
            count += 1
    return count


def export_jsonl(chapters, path, min_len):
    """导出指令格式：供微调训练"""
    count = 0
    chapter_re = re.compile(r"^第[零一二三四五六七八九十百千万0-9]+[章回节卷]")
    with open(path, "w", encoding="utf-8") as f:
        for title, body in chapters:
            if len(body) < min_len:
                continue
            if chapter_re.match(title):
                instruction = f"根据标题《{title}》写一篇玄幻小说章节"
            else:
                instruction = "续写下面的玄幻小说内容"
            record = {"instruction": instruction, "output": body}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    if len(sys.argv) < 3:
        print("用法: py clean_novel.py 输入文件.txt 输出名 [最低保留字数]")
        print("示例: py clean_novel.py 斗破苍穹.txt doupo 300")
        sys.exit(1)

    src, out_name = sys.argv[1], sys.argv[2]
    min_len = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    text = read_novel(src)
    lines = clean_lines(text)
    chapters = split_chapters(lines)

    txt_path = out_name + ".txt"
    jsonl_path = out_name + ".jsonl"
    n1 = export_txt(chapters, txt_path, min_len)
    n2 = export_jsonl(chapters, jsonl_path, min_len)

    print(f"\n导出完成:")
    print(f"  {txt_path}   <- {n1} 章（供 BPE tokenizer 训练）")
    print(f"  {jsonl_path} <- {n2} 章（供微调训练）")


if __name__ == "__main__":
    main()
