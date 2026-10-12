# data/sample —— 格式样例

这里放的是**用于说明数据格式的一小段自写文本**，不是任何真实语料。

本仓库不提交小说原文，原因是体积与版权（见根目录 README「关于语料」一节）。
`data/raw/` 与 `data/processed/` 均已在 `.gitignore` 中，只保留脚本与样例。

## 语料该长什么样

`scripts/clean_novel.py` 的输入是**纯文本小说文件**，要求：

1. 章节标题单独成行，以 `第X章` / `第X回` / `第X节` / `第X卷` 开头（X 可以是中文数字或阿拉伯数字）；
2. 正文与标题同一编码，UTF-8 / GBK / UTF-16 均可，脚本会自动识别；
3. 广告行、网址、站点名等噪音可以不用提前清理，脚本会处理。

输出两份文件：

- `<名字>.txt` —— 按章拼接的纯文本，章节之间空行分隔，供分词器训练使用；
- `<名字>.jsonl` —— 每行一条 `{"instruction": ..., "output": ...}`，供指令微调使用。

## 一个最小例子

见 [example_format.txt](example_format.txt)。把语料放进 `data/raw/` 后运行：

```bash
py scripts/clean_novel.py data/raw/你的小说.txt data/processed/你的小说 300
```
