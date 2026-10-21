# MiniQwen

从零训练一个**中文小说创作**小模型。全链路自实现：数据清洗 → 字节级 BPE 分词器 → Qwen2 风格 Decoder → 预训练 → 指令微调 → 采样推理。

> 目标不是"训出一个能替代 Qwen 的模型"，而是**把大模型训练的每一个环节都亲手实现一遍，并留下可复现的实验记录**。

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境、仓库骨架、GPU 可用 | ✅ 完成 |
| 1 | 数据管线（清洗 / 切章 / 去重 / 统计报告） | ✅ 完成 |
| 2 | Tokenizer 定稿（16k 词表 + 特殊 token + 压缩率评测） | ✅ 完成 |
| 3 | 二进制语料（`uint16` memmap，训练时内存零占用） | ✅ 完成 |
| 4 | 模型实现（RMSNorm / RoPE / SwiGLU / GQA）+ 与 Qwen2.5-0.5B 权重对齐验证 | 🚧 进行中 |
| 5 | 预训练（bf16 + 梯度累积 + 断点续训 + loss 曲线） | ⬜ |
| 6 | SFT 指令微调 + KV Cache 推理 + Gradio demo | ⬜ |
| 7 | 评测、消融实验、CI、文档 | ⬜ |

完整行动计划见 [ROADMAP.md](ROADMAP.md)，清洗阶段的踩坑记录见 [reports/cleaning_notes.md](reports/cleaning_notes.md)。

### 已完成的关键结果

- **数据**：`斗破苍穹` 10.8 MB / GBK → 清洗出 **1,586 章 / 520.7 万字**，1.5 秒跑完
- **水印清洗**：精确删除 **40,494 处**「武动乾坤」追踪水印 + 各类站点广告，残留 0
  （旧脚本残留 200 处，且无差别砍掉 6,862 行正常句尾）
- **发现并修复** 3 个数据缺陷：短段落被误删、标题前缀误判导致整章丢失、URL 正则会吃掉后续正文
- **分词器**：字节级 BPE，词表 16,384，压缩率 **5.35 字节/token（1.81 字/token）**，
  588 万字节的留出文本编成 11 万 token，往返无损；`萧炎点了点头，淡淡的道：` → 6 个 token
- **二进制语料**：**2,869,937** 个 token 固化成 `uint16`，共 **5.47 MB**
  （比 14.77 MB 的原始文本还小）；切分点对齐到章节边界；
  随机取 1000 个 512 长窗口：从 id 数组 2.41 ms，从文本重新分词约 1.1 秒（**447 倍**）
- **测试**：74 passed（含 5 个记录已知缺陷的 xfail）

## 环境安装

```bash
# 1) 装 CUDA 版 torch（关键！Windows 上 PyPI 默认是 CPU 版）
py -m pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130

# 2) 装其余依赖
py -m pip install -r requirements.txt

# 3) 验证 GPU 可用
py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

已验证环境：Python 3.12 / PyTorch 2.14.1+cu130 / RTX 5060 Laptop 8GB（Blackwell, sm_120）/ Windows。

> RTX 50 系是 Blackwell 架构（sm_120），需要 **CUDA 12.8 以上**的 torch 轮子，否则会出现
> `no kernel image is available for execution on the device`。

## 目录结构

```
miniqwen/          核心包：模型、分词器、数据集、训练、推理
scripts/           数据处理脚本（清洗 / 分词器训练 / 语料准备）
configs/           模型与训练配置
tests/             单元测试
reports/           实验报告：数据统计、分词器评测、loss 曲线、消融实验
data/              语料与分词器（原文与处理产物不提交，见下）
```

## 关于语料

**本仓库不包含任何小说原文。** 原因有两个：体积（数百 MB 不适合放 Git）和版权。

- `data/raw/`、`data/processed/` 已在 `.gitignore` 中；
- 仓库只保留数据处理脚本、一小段自写的格式样例（`data/sample/`）和训练好的分词器；
- 如需复现，请自行准备**公有领域或已获授权**的中文文本，放入 `data/raw/` 后运行 `scripts/clean_novel.py`。

## License

[MIT](LICENSE)
