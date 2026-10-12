# MiniQwen 行动脉络（从零训练一个中文小说创作小模型）

> 版本 v1.0 ｜ 适用者：985 计科大二 ｜ 目标：一个**可复现、有量化对比、能讲清楚每个设计决策**的开源项目
>
> 本文件是总纲。每完成一个阶段，回来把该阶段的 `[ ]` 打勾，并同步 `README.md` 的进度条。

---

## 一、项目最终形态（先想清楚终点，再倒推）

一个仓库，跑完下面这条链路：

```
原始小说 txt
  → 清洗 / 切章 / 去重          （已有 v4 版）
  → 字节级 BPE 分词器（带特殊 token）
  → uint16 二进制语料（可 memmap，训练时不占内存）
  → Qwen2 风格 Decoder 模型（RMSNorm + RoPE + SwiGLU + GQA）
  → 预训练（bf16 + 梯度累积 + 断点续训 + loss 曲线）
  → 指令微调 SFT（chat template）
  → 采样推理（KV Cache / top-k / top-p / 重复惩罚）
  → CLI 对话 + Gradio 网页 demo
```

最终产物：

| 产物 | 说明 | 放哪 |
|---|---|---|
| 代码仓库 | 全部训练/推理代码 + 单测 + CI | GitHub |
| 模型权重 | 预训练 + SFT 权重（几十~几百 MB） | ModelScope / HuggingFace（**不放 GitHub**） |
| `reports/` | loss 曲线、数据报告、样例对比、消融实验表 | GitHub |
| 技术博客 | 3~5 篇踩坑记录 | 知乎 / CSDN / 个人博客 |

**这个项目最大的价值不是"训出一个好模型"，而是"你能把每个环节的为什么讲清楚"。**
评委/面试官不会去看你的 loss 是 2.8 还是 3.2，他们会问：为什么用 RoPE 不用可学习位置编码？vocab 为什么选 16k？显存爆了怎么解决？loss 不降你怎么办？

---

## 二、现状盘点（2026-09-25 实测）

### 已完成
| 文件 | 内容 | 状态 |
|---|---|---|
| `clean_novel.py` | 三层清洗（整行删除 / 行内切除 / 高频后缀剥离）+ 章节切分 + 双格式导出 | ✅ 质量不错，可以进仓库 |
| `斗破苍穹.txt` | 原始语料 11.3 MB | ✅ |
| `doupo.txt` | 清洗后正文 **528 万字 / 1547 万字节 / 1500 章** | ✅ |
| `doupo.jsonl` | 指令格式 `{instruction, output}` **1389 条** | ✅ |
| `bpe_tokenizer.py` | 从零实现字节级 BPE，**增量统计优化 + 留出法评测** | ✅ 亮点，但见下方卡点 |
| `tokenizer.json` | 4744 次合并 / 词表 5000 | ⚠️ 需重训 |

### 环境实测
| 项 | 实测结果 | 结论 |
|---|---|---|
| GPU | RTX 5060 Laptop，**8 GB**，驱动 592.01 | Blackwell 架构（sm_120），**必须 CUDA 12.8+ 的 torch** |
| PyTorch | `2.14.0+cpu` | ❌ **CPU 版，GPU 完全用不上**，必须换 cu130 版 |
| Python | 3.12.10（用户目录安装） | ✅ 可用；但 `python` 命令被微软商店别名劫持，要用 `py` |
| 已装包 | numpy 2.5.3 / tqdm 4.70.1 | 其余全缺 |
| git | 2.56.0 | ✅ 但**未配置 user.name / user.email**，无法提交 |
| 网络 | 清华 PyPI ✅ / download.pytorch.org ✅ / ModelScope ✅ / **github.com ❌ 超时** / api.github.com ✅ / ssh.github.com:443 ✅ | 见 §六 Git 通道方案 |

### 三个必须解决的卡点
1. **torch 是 CPU 版** → 32 万倍的时间差，必须先换。
2. **`bpe_tokenizer.py` 的增量统计有实测缺陷（2026-09-25 已验证）**：
   `merge_pair_incremental()` 在「相邻位置同时发生合并」时会把中间那个 pair 重复计数——
   一侧多做一次减法、另一侧多做一次加法。实测：对 `[1,1,1,2,1,2,1,2,3]` 做 merge `(1,2)` 后，
   pair `(2,1)` 的计数变成 **-2**（真实值 0），而新 pair `(256,256)` 变成 **4**（真实值 2）。
   后果：合并顺序次优、压缩率偏低、`best_cnt < 2` 可能提前终止——**而且不报错**，
   分词器照样能训完。回归测试已写入 `tests/test_bpe_counts.py`。
   同时 `encode()` 对每段文本做 4744 次全量扫描，编码 528 万字在纯 Python 下不可完成。
   这两点共同决定了阶段 2 的技术路线（见决策 3）。
3. **没有特殊 token**：`<|im_start|>` 这类模板 token 缺失，SFT 阶段无法做对话格式。

---

## 三、关键决策（现在定下来，后面不返工）

### 决策 1：模型结构走 Qwen2 路线，不走 minGPT 的 GPT-2 路线
`karpathy/minGPT` 是 2018 年的 GPT-1/2 结构（可学习位置编码 + GELU + LayerNorm）。
你既然要"参考千问"，就应该实现 **Qwen2 结构**：

| 组件 | GPT-2（minGPT） | **Qwen2（本项目）** |
|---|---|---|
| 归一化 | LayerNorm | **RMSNorm** |
| 位置编码 | 可学习绝对位置 | **RoPE（旋转位置编码）** |
| FFN | GELU MLP | **SwiGLU** |
| 注意力 | MHA | **GQA（分组查询注意力）** |
| 偏置 | 全有 | Linear 无 bias，**但 QKV 有 bias**（Qwen2 特征） |
| 词嵌入 | 独立输出层 | **权重绑定（tied）** |

minGPT 只借它的"参数规模档位"思路（nano/mini/small），结构按 Qwen2 来。
好处：面试聊的全是当前主流技术；而且**可以把官方 Qwen2.5-0.5B 的权重加载进你自己写的模型**做对齐验证（阶段 3 的杀手锏）。

### 决策 2：模型规模与语料规模（先算清楚账，别瞎调大）
- 单本斗破 ≈ **530 万 token**。
- 训练算力：`FLOPs ≈ 6 × 参数量 × token数`。30M 参数跑 1 个 epoch ≈ `9.5e14` FLOPs，在 5060 上约 **1~3 分钟/epoch**。所以**算力不是瓶颈，数据才是**。
- Chinchilla 最优配比 ≈ 20 token/参数 → 530 万 token 对应只有 26 万参数，说明**单本语料严重不足**，10M+ 参数必然过拟合。

**结论（分两步走，别一次到位）：**
- **v0 跑通版**：单本斗破，10~30M 参数，训 3~5 epoch。目标是**把管线全部打通**，接受输出质量一般。
- **v1 正式版**：语料扩到 **10~20 本合规小说（3000 万~1 亿字）**，模型 60~100M 参数。这才是拿得出手的版本。

8 GB 显存预算（100M 参数 / bf16 / AdamW）：权重 0.2G + 梯度 0.2G + 优化器状态 1.2G + 激活 ≈ 2~3G，**总计 4~5 GB，够用**。超了就用梯度累积 + 梯度检查点。

### 决策 3：Tokenizer 走"双轨制"（这是本阶段最重要的决定）
| | 路线 A：双轨（**推荐**） | 路线 B：纯从零 |
|---|---|---|
| 教学版 | 保留你现在的 `bpe_from_scratch.py`，在小样本（20 万字）上训练 + 可视化合并过程 | 同左 |
| 生产版 | 用 HF `tokenizers`（Rust 实现）在 528 万字全量上训练同算法 BPE，vocab 16384 + 特殊 token | 自己重写 `encode()`（minbpe 式：反复合并 rank 最小的 pair）+ 正则预切分 + 多进程并行 |
| 优点 | 快（分钟级）、可靠、有特殊 token；README 里"我理解了算法所以我知道生产版怎么实现"是加分项 | 血统纯正，全部自己写 |
| 缺点 | 多一个依赖 | 慢，且容易在性能上耗掉一周 |

**推荐 A。** 你的从零版已经是很好的作品了，不需要在工程性能上反复造轮子——把时间花在模型和训练上。
若坚持 B，务必加正则预切分（`re` 按中文字符/数字/标点切块），否则合并会跨越标点边界。

### 决策 4：语料合规（**重要，关系到项目能不能公开**）
- 斗破苍穹是**有版权的商业作品**。自己学习没问题，但**把 15 MB 原文提交到公开仓库 = 侵权风险**，也可能被 DMCA 删库。
- 所以：`data/` 全部进 `.gitignore`，仓库里只放**清洗脚本 + 下载说明 + 一小段样例（一两章）**。
- 扩语料请用**公有领域 / 已获授权 / 自己创作**的文本（古龙金庸也不可公开分发；《红楼梦》《西游记》《三国演义》等古典名著可放心用）。

---

## 四、分阶段行动脉络

> 时间按"每天 3~4 小时"估算。总共 **4~6 周**。每个阶段末尾都有一个 **git 提交 + 验收标准**，验收不过不进入下一阶段。

### 阶段 0：环境地基与第一次 push（0.5~1 天）

**目标**：GPU 能跑、仓库干净、代码成功推到 GitHub。

任务：
- [ ] 装 CUDA 版 PyTorch：`torch==2.14.1+cu130`（RTX 5060 是 Blackwell sm_120，CUDA 12.6 以下的轮子认不出这张卡）
- [ ] 验证：`torch.cuda.is_available()` → True，且能打印出 "RTX 5060"
- [ ] 建立一个真实计算测试（矩阵乘 + 反向传播），确认不是"能 import 但不能算"
- [ ] 配置 git 身份：`git config --global user.name/user.email`
- [ ] 建仓库骨架（见 §五 目录结构），写 `.gitignore` / `README.md` / `LICENSE(MIT)` / `requirements.txt`
- [ ] `git init` → 首次提交 → 建 GitHub 仓库 → `git push`

**验收**：GitHub 网页上能看到仓库，`git log` 有第一条提交，`python -c "import torch;print(torch.cuda.get_device_name(0))"` 输出显卡型号。
**提交**：`chore: 初始化项目骨架与依赖`

---

### 阶段 1：数据管线工程化（2~3 天）✅ 已完成

**目标**：把"一个脚本处理一个文件"升级成"一条可复现的数据流水线"。

任务：
- [x] `clean_novel.py` 重构进 `miniqwen/corpus.py`（纯函数库）+ `scripts/build_corpus.py`（CLI）：
      支持**目录批处理**、多本小说、自动编码识别、清洗日志
- [x] 三层清洗规则全部换成有真实数据证据支撑的精确规则（详见 `reports/cleaning_notes.md`）
- [x] `reports/data_report.md`：清洗账本 + **规则命中分解**（自动生成）
- [x] `reports/cleaning_notes.md`：清洗踩坑记录与量化对比（手写）
- [x] `tests/fixtures/` 测试样例 + `tests/test_corpus.py`
- [x] 单遍流式处理：内存占用与语料总量无关
- [x] `data/sample/` 放一小段可公开的样例（自己写的）
- [ ] 扩语料：收集 10~20 本**合规**小说（**推迟到 v1**，v0 先单本跑通）

> `prepare_data.py`（文本 → `uint16` 二进制语料）**推迟到阶段 2 之后**：
> 它需要先有分词器才能确定词表大小和 dtype，放在阶段 1 是顺序错误。

**实测结果**：1.2 秒处理 528 万字 → 1,586 章 / 1582 条 SFT 记录；
精确删除 40,494 处追踪水印，零残留；相比旧脚本多出 14% 可用数据。
**提交**：`feat(data): 批处理语料流水线 + 精确水印清洗 + 数据报告`

> **为什么值得花 3 天**：大厂真正缺的是数据工程能力。面试问"你怎么处理脏数据"时，你能给出三层规则 + 去重率 + 统计报告，这一条就赢过 90% 的候选人。

---

### 阶段 2：Tokenizer 定稿（2~3 天）

**目标**：一个**能编码 500 万字**、带特殊 token、有量化评测的分词器。

任务：
- [ ] 把现有 `bpe_tokenizer.py` 移到 `scripts/bpe_from_scratch.py`（保留 + 重命名为"教学版"，加注释说明增量统计的原理）
- [ ] 生产版 `scripts/train_tokenizer.py`：vocab **16384**、正则预切分、特殊 token `<|pad|> <|unk|> <|bos|> <|eos|> <|im_start|> <|im_end|>`
- [ ] `miniqwen/tokenizer.py`：封装 `encode / decode / apply_chat_template`，统一接口给训练和推理用
- [ ] 评测表 `reports/tokenizer_report.md`：
  - 压缩率（bytes/token、字符/token）
  - 编解码往返**无损率**（必须 100%）
  - 一章（3000 字）平均多少 token
  - vocab 5000 vs 16384 的对比
- [ ] 单测 `tests/test_tokenizer.py`：任意中英混合文本往返一致、特殊 token 不被打碎

**验收**：全量 528 万字能在 5 分钟内编码完；往返无损 100%；报告里有对比表格。
**提交**：`feat(tokenizer): 字节级BPE定稿(16384词表+特殊token)`、`test(tokenizer): 往返无损与特殊token测试`

---

### 阶段 3：模型实现（3~5 天）

**目标**：一个**正确**的 Qwen2 风格 Decoder，且有单元测试证明它正确。

任务：
- [ ] `miniqwen/model.py`：
  - `RMSNorm`
  - `RoPE`（预计算 cos/sin 缓存，支持 `theta=10000` 和 `rope_scaling`）
  - `SwiGLU` FFN（`intermediate = 8/3 * d_model` 向上取整到 64 的倍数，这是 Qwen 的做法）
  - `GQA` 注意力（`num_kv_heads < num_heads`，用 `repeat_interleave` 展开）
  - **因果掩码**（用 `torch.triu` 或 `is_causal=True`，二选一但要写清楚）
  - `Qwen2ForCausalLM`：tied embedding、QKV bias、输出 logits
- [ ] `miniqwen/config.py` + `configs/*.json`：`nano`(≈10M) / `mini`(≈30M) / `small`(≈100M)
- [ ] `tests/test_model.py`：
  - 形状测试（各种 batch/seq 组合，含 seq=1）
  - **因果性测试**：改后面一个 token，前面的 logits 必须不变（这是最容易写错的地方）
  - 参数量与预估一致
  - 前向 + 反向能跑通，梯度非 NaN
- [ ] **杀手锏（强烈建议做）**：从 ModelScope 下 `Qwen2.5-0.5B`，把官方权重加载进**你自己写的模型**，对比同一输入的 logits 误差 < 1e-4。做成了写进 README，这是简历上最硬的一条。

**验收**：`pytest tests/` 全绿；对齐验证 logits 误差达标；`print(model)` 参数量与手算一致。
**提交**：`feat(model): Qwen2风格解码器(RMSNorm/RoPE/SwiGLU/GQA)`、`test(model): 因果性与形状测试`、`feat(model): 与Qwen2.5-0.5B官方权重对齐验证`

---

### 阶段 4：预训练（5~10 天，含调参）

**目标**：loss 稳定下降，产出可用的 base 模型。

任务：
- [ ] **先做单 batch 过拟合**：拿 1 个 batch 反复训，loss 必须降到接近 0。不降就说明代码有 bug，先别开训（这一步能省掉你一周）
- [ ] `miniqwen/train.py`：
  - AdamW（`betas=(0.9,0.95)`、`weight_decay=0.1`，**norm 和 bias 不衰减**）
  - 余弦退火 + warmup（前 2% 步数）
  - 梯度裁剪 1.0
  - `torch.autocast(bf16)` + `GradScaler`（bf16 通常不需要 scaler）
  - 梯度累积（显存不够时的救命稻草）
  - 梯度检查点（可选，省显存换时间）
  - **断点续训**（被笔记本合盖打断是常态，这个必须有）
  - 日志：每 N 步记 loss/lr/显存峰值/tokens per second
  - 定期在 val 集上算 loss + 生成样例（看文字，别只看数字）
- [ ] 画 loss 曲线（`reports/loss_curve.png`）：train vs val，判断过拟合点
- [ ] 记录性能数字：**tokens/s、显存峰值、单 epoch 耗时**（这三个数字写进 README）

**v0（单本）→ v1（扩语料）各训一次，做对比。**

**验收**：val loss 明显低于随机基线（ln(16384)≈9.7）；生成样例是通顺中文；有曲线图。
**提交**：`feat(train): 预训练循环(bf16+梯度累积+断点续训)`、`chore(ckpt): v0.1 预训练权重`、`docs: 预训练实验记录与loss曲线`

---

### 阶段 5：SFT 指令微调 + 推理（3~5 天）

**目标**：模型能"按指令写小说"，且有交互界面。

任务：
- [ ] `miniqwen/sft.py`：用 `doupo.jsonl`，套 chat template：
  ```
  <|im_start|>user
  根据标题《第一章 陨落的天才》写一篇玄幻小说章节<|im_end|>
  <|im_start|>assistant
  ...<|im_end|>
  ```
  **loss 只在 assistant 部分计算**（这是 SFT 的关键细节，很多人做错）
- [ ] `miniqwen/generate.py`：temperature / top-k / top-p / 重复惩罚 / **KV Cache**
- [ ] **KV Cache 的加速比要量化**：写进 README（例如 "有缓存 180 tok/s，无缓存 22 tok/s，8.2×"）
- [ ] `miniqwen/chat.py`：命令行交互
- [ ] `app.py`：Gradio 网页 demo（截图进 README）

**验收**：输入标题能生成一章；CLI 能多轮对话；demo 能跑起来。
**提交**：`feat(sft): 指令微调(chat template+loss mask)`、`feat(inference): KV Cache与采样策略`、`feat(demo): Gradio 网页演示`

---

### 阶段 6：评测、工程化与发布（3~5 天）

**目标**：把这个仓库变成"别人能复现、面试官愿意读"的作品。

任务：
- [ ] 评测 `reports/evaluation.md`：困惑度、生成样例网格、**消融实验表**（vocab 5k vs 16k、有无 RoPE、不同 lr）
- [ ] `pytest` 补全 + `.github/workflows/ci.yml`（CPU 上跑单测，需要能访问 GitHub Actions）
- [ ] `requirements.txt` 锁版本 + `ruff` 格式化 + 类型注解 + docstring
- [ ] README 大改：架构图、loss 曲线、生成样例对比、**复现步骤（三条命令）**、性能数字表
- [ ] 权重上传 ModelScope（顺便放一份到 HuggingFace）
- [ ] 写 3~5 篇技术博客：①从零实现BPE ②RoPE 到底在转什么 ③8GB 显存训 100M 模型 ④为什么我的 loss 不下降 ⑤KV Cache 为什么快
- [ ] 打 tag：`v1.0.0`，写 Release Notes

**提交**：`docs: 评测报告与消融实验`、`ci: GitHub Actions 单测`、`docs: README 重写`
**Tag**：`v1.0.0`

---

## 五、目录结构（阶段 0 建好，后面往里填）

```
mini_qwen/
├── README.md                   # 门面：架构图/曲线/样例/复现步骤
├── ROADMAP.md                  # 本文件
├── LICENSE                     # MIT
├── requirements.txt
├── .gitignore                  # data/ checkpoints/ .venv/ __pycache__/
├── configs/
│   ├── model_nano.json         # ≈10M
│   ├── model_mini.json         # ≈30M
│   ├── model_small.json        # ≈100M
│   ├── train_pretrain.json
│   └── train_sft.json
├── miniqwen/                   # 核心包
│   ├── __init__.py
│   ├── config.py               # 配置 dataclass
│   ├── model.py                # ★ Qwen2 结构
│   ├── tokenizer.py            # 封装 + chat template
│   ├── dataset.py              # memmap 数据集
│   ├── train.py                # ★ 预训练
│   ├── sft.py                  # 指令微调
│   ├── generate.py             # ★ 采样 + KV Cache
│   └── utils.py
├── scripts/
│   ├── clean_novel.py          # 已有（升级为批处理）
│   ├── bpe_from_scratch.py     # 已有（教学版，保留）
│   ├── train_tokenizer.py      # 生产版分词器
│   └── prepare_data.py         # 语料 → .bin
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── test_pipeline.py        # 单 batch 过拟合
├── data/                       # ★ 全部 gitignore
│   ├── raw/                    # 原始小说
│   ├── processed/              # doupo.txt / .jsonl / .bin
│   ├── tokenizer/              # tokenizer.json（小，可提交）
│   └── sample/                 # 可公开的样例（1 章，可提交）
├── checkpoints/                # ★ gitignore
├── reports/
│   ├── data_report.md
│   ├── tokenizer_report.md
│   ├── evaluation.md
│   └── loss_curve.png
├── app.py                      # Gradio demo
└── .github/workflows/ci.yml
```

---

## 六、Git 通道方案（本机 github.com 被墙，实测结论）

| 通道 | 实测 | 能否用 |
|---|---|---|
| `https://github.com` | 超时（DNS 解析到 140.82.121.4，被阻断） | ❌ |
| 其他 GitHub IP（140.82.112.3 / 20.205.243.166） | TCP+TLS 通，但慢（15~30 s） | ⚠️ 需改 hosts，要管理员权限 |
| `https://api.github.com` | **419 ms，正常** | ✅ 可用 API 建仓库、加公钥 |
| `ssh.github.com:443` | **握手成功（返回 Permission denied (publickey)）** | ✅ **主方案** |
| `https://gh-proxy.com/https://github.com/...` | 200 正常 | ✅ 用来 clone 别人的参考仓库 |
| ModelScope / hf-mirror.com | 正常 | ✅ 下模型和数据集 |

**主方案：SSH over 443**（不需要管理员权限）
1. `ssh-keygen -t ed25519` 生成密钥
2. 把公钥加到 GitHub 账号（浏览器登录一次 GitHub 网页；或用 PAT + `api.github.com`，我可以帮你调 API）
3. `~/.ssh/config` 写入：
   ```
   Host github.com
       Hostname ssh.github.com
       Port 443
       User git
   ```
4. `git remote add origin git@github.com:<你的用户名>/miniqwen.git`

**参考仓库怎么拉**（github.com 直连不了）：
```
git clone https://gh-proxy.com/https://github.com/karpathy/nanoGPT
git clone https://gh-proxy.com/https://github.com/karpathy/minbpe
git clone https://gh-proxy.com/https://github.com/jingyaogong/minimind
```
Qwen2 官方建模代码从 ModelScope 拿。

### 提交规范
- Conventional Commits：`feat: / fix: / docs: / test: / chore: / refactor:`
- 中文描述也完全可以，保持一致即可（推荐 `<type>: 中文说明`）
- 每个阶段打一个 tag：`v0.1-env` → `v0.2-data` → `v0.3-tokenizer` → `v0.4-model` → `v0.5-pretrain` → `v0.6-sft` → `v1.0.0`
- **不要提交**：`data/raw/`、`data/processed/`、`checkpoints/`、`.venv/`、`__pycache__/`、`.ipynb_checkpoints/`
- 提交前 `git status` 看一眼，第一条命令学会 `git status` 而不是 `git add .`

---

## 七、要装什么（按必要性排序）

| 包 | 必要性 | 装在哪一步 | 说明 |
|---|---|---|---|
| `torch==2.14.1+cu130` | **必须** | 阶段 0 | 当前是 CPU 版，必须替换。约 3 GB 下载 |
| `numpy` `tqdm` | 已有 | — | |
| `tokenizers` | 推荐 | 阶段 2 | Rust 实现的 BPE，生产版分词器用 |
| `matplotlib` | 推荐 | 阶段 4 | 画 loss 曲线 |
| `pytest` | 推荐 | 阶段 3 | 单测 |
| `gradio` | 可选 | 阶段 5 | 网页 demo，简历上很加分 |
| `ruff` | 可选 | 阶段 6 | 代码风格 |
| `transformers` + `safetensors` | 可选 | 阶段 3 | 仅在做 Qwen 权重对齐验证时需要 |
| `wandb` | **不建议** | — | 服务器在国外，慢且要账号；用 TensorBoard 或自己写 CSV + matplotlib |
| `flash-attn` | **不建议** | — | Windows 上编译地狱；`torch.nn.functional.scaled_dot_product_attention`（SDPA）已经够快 |

---

## 八、风险清单（提前知道就不会慌）

| 风险 | 表现 | 对策 |
|---|---|---|
| **数据量不足** | val loss 先降后升，生成内容复读 | 扩语料；加 dropout / weight decay；减少 epoch；早停 |
| **显存 OOM** | 训练中途崩 | 降 batch、上梯度累积、开梯度检查点、缩短 seq_len |
| **loss 不下降** | 一直在 9.7 附近 | 先做单 batch 过拟合；检查 lr（太小）、mask 写反、label 对齐错位（shift 错一位） |
| **loss 变 NaN** | 突然爆 | 梯度裁剪、降 lr、warmup 加长、检查除零 |
| **版权** | 仓库被封 | data 全部 gitignore，只提交脚本和样例 |
| **学到一半跑去调参** | 时间全花在超参上 | 阶段 4 先用固定超参跑通完整一轮，再谈调优 |
| **笔记本过热/合盖中断** | 训练白跑 | 断点续训 + 每 N 步存 ckpt（阶段 4 必做） |

---

## 九、这件事对保研 / 实习的价值（说实话）

### 保研：**加分项，但不是决定项**
- 保研的核心权重是：**GPA/专业排名 > 机试/笔试 > 面试表现 > 科研经历 > 项目经历**。
- 这个项目的真实作用是：
  1. **个人陈述/简历里能写出一个完整的技术故事**（不是"用过 PyTorch"，而是"我实现并训练过一个 LLM"）。
  2. **面试时的"防空洞"**：老师问"你了解 Transformer 吗"，你可以从 RoPE 的旋转矩阵讲到 GQA 省了多少 KV 显存——大部分同学只能背八股。
  3. 如果做扎实了，这可以**变成进实验室的敲门砖**：拿这套代码去找做 NLP/大模型的老师，说"我复现过 Qwen2 结构并训练过小模型，想在您组里做 XX"，比空手去强太多。
- **它不能替代的**：一篇论文/一段真正的科研训练。如果你目标是顶尖组，项目只是"证明你能干活"，科研经历才是"证明你能出成果"。
- **最好的用法**：把这个项目 + 3 篇技术博客 + ModelScope 上的权重链接，一起放进 GitHub Profile README，作为你的技术名片。

### 实习：**价值更大，尤其是算法/工程岗**
- 大厂 LLM 相关实习面试，最常见的灵魂拷问就是：
  - "你训过最大的模型是多大？显存怎么优化的？"（→ 你有 tokens/s、显存峰值数字）
  - "loss 不收敛你怎么排查？"（→ 你有"单 batch 过拟合"这个真实经验）
  - "tokenizer 怎么选的？词表大小有什么影响？"（→ 你做 5k vs 16k 对比实验）
  - "了解 KV Cache 吗？"（→ 你有 8× 加速的实测数据）
- 这些问题，**跑通教程的人和真正踩过坑的人，回答质量差十倍**。
- 但要注意：**头部大模型团队（如做预训练/后训练的）通常还看论文和竞赛**。而这个项目非常适合投：**数据工程、训练框架、推理优化、模型评测、AI 应用** 这些方向的实习——它们恰恰最看重动手能力。
- 简历上请这样写（要数字，不要形容词）：
  > **MiniQwen｜中文小说生成小模型（个人项目）**
  > 从零实现字节级 BPE 分词器（16k 词表，压缩率 x.x bytes/token）；实现 Qwen2 风格 Decoder（RMSNorm/RoPE/SwiGLU/GQA），与官方 Qwen2.5-0.5B 权重对齐误差 <1e-4；在 RTX 5060 8GB 上训练 xxM 参数模型，通过梯度累积+检查点将显存降至 x.x GB，吞吐 xx tokens/s；实现 KV Cache 使推理提速 x.x 倍；SFT 微调后支持按标题生成章节，Gradio demo 已部署。

### 三个"让价值翻倍"的动作（比多写代码更重要）
1. **量化一切**：每个优化都留下前后对比数字。没有数字的优化等于没做。
2. **写下来**：3~5 篇博客。技术写作能力在大厂是稀缺资源，也是面试官判断你"能不能带团队/写文档"的依据。
3. **讲得清**：反复练习 3 分钟讲完这个项目——为什么做、做了什么、最难的点、怎么解决的、有什么不足。**能讲清楚"不足"的人，比只会吹的人可信得多。**

### 一句提醒
**不要为了"项目看起来大"而堆功能。** 一个 100M 模型 + 完整评测 + 清楚文档的仓库，远胜过一个"什么都有但都跑不通"的仓库。宁缺毋滥。

---

## 十、时间线速览

| 周次 | 阶段 | 关键产出 |
|---|---|---|
| 第 1 周 | 阶段 0 + 1 | 环境 + 仓库上线 + 数据管线 |
| 第 2 周 | 阶段 2 + 3 | 分词器定稿 + 模型实现与对齐验证 |
| 第 3 周 | 阶段 4 | 预训练跑通（v0）+ loss 曲线 |
| 第 4 周 | 阶段 4（v1）+ 5 | 扩语料重训 + SFT + 推理 |
| 第 5 周 | 阶段 6 | 评测 + CI + 文档 + 博客 |
| 第 6 周 | 缓冲 | 补实验、修 README、准备答辩式讲解 |

---

**下一步**：从阶段 0 开始。装 CUDA 版 torch → 建骨架 → 第一次 push。
