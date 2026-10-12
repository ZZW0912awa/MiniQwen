# MiniQwen

从零训练一个**中文小说创作**小模型。全链路自实现：数据清洗 → 字节级 BPE 分词器 → Qwen2 风格 Decoder → 预训练 → 指令微调 → 采样推理。

> 目标不是"训出一个能替代 Qwen 的模型"，而是**把大模型训练的每一个环节都亲手实现一遍，并留下可复现的实验记录**。

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境、仓库骨架、GPU 可用 | 🚧 进行中 |
| 1 | 数据管线（清洗 / 切章 / 去重 / 二进制语料） | ⬜ |
| 2 | Tokenizer 定稿（16k 词表 + 特殊 token + 压缩率评测） | ⬜ |
| 3 | 模型实现（RMSNorm / RoPE / SwiGLU / GQA）+ 与 Qwen2.5-0.5B 权重对齐验证 | ⬜ |
| 4 | 预训练（bf16 + 梯度累积 + 断点续训 + loss 曲线） | ⬜ |
| 5 | SFT 指令微调 + KV Cache 推理 + Gradio demo | ⬜ |
| 6 | 评测、消融实验、CI、文档 | ⬜ |

完整行动计划见 [ROADMAP.md](ROADMAP.md)。

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
