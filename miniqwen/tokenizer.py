# -*- coding: utf-8 -*-
"""分词器封装：加载 tokenizer.json、编码/解码、套对话模板。

为什么单独包一层
----------------
底层的 tokenizers 库和训练脚本都不该被上层代码直接 import。
把「特殊 token 的名字」「文件放哪」「对话模板长什么样」全部收在这个文件里，
以后要换成别的分词器实现，只需要改这一个文件。

特殊 token 的作用（预训练时用不到，SFT 阶段必须）
--------------------------------------------------
<|im_start|> / <|im_end|>  把对话切成「谁说的、说到哪」，模型才知道该在哪停
<|pad|>                     批量训练时把短样本补齐到同一长度（要 mask 掉，不参与 loss）
<|bos|> / <|eos|>           序列的开头 / 结尾
<|unk|>                     兜底（字节级 BPE 其实用不到，但留着防止以后换实现）

注意：这些 token 必须【在训练分词器时就注册进去】，不能用完再补。
后补的话它们的 id 是新的，预训练好的 embedding 里没有对应行。
"""

from __future__ import annotations

from pathlib import Path

# ============================================================
# 特殊 token —— 唯一真相源，训练脚本也从这里 import
# ============================================================
SPECIAL_TOKENS: list[str] = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|im_start|>",
    "<|im_end|>",
]
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, IM_START, IM_END = SPECIAL_TOKENS

# 默认位置：仓库根/data/tokenizer/tokenizer.json
DEFAULT_TOKENIZER_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "tokenizer" / "tokenizer.json"
)


class NovelTokenizer:
    """中文小说分词器的薄封装。

    用法:
        tok = NovelTokenizer()
        ids = tok.encode("萧炎点了点头。")          # 预训练用，不带特殊 token
        tok.decode(ids)                             # 还原成文本
        text = tok.apply_chat_template("写一段斗气突破", "萧炎盘膝而坐……")  # SFT 用
    """

    def __init__(self, path: str | Path = DEFAULT_TOKENIZER_PATH):
        # 延迟 import：只在真要建对象时才加载 tokenizers 库，
        # 这样别处 import 本模块（比如拿 SPECIAL_TOKENS）不会连带拖慢启动。
        from tokenizers import Tokenizer

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"找不到分词器文件: {self.path}\n"
                f"先跑：py scripts/train_tokenizer.py"
            )
        self._tok = Tokenizer.from_file(str(self.path))

        # 缓存几个常用 id，避免每次去词表里查
        self.pad_id = self._tok.token_to_id(PAD_TOKEN)
        self.unk_id = self._tok.token_to_id(UNK_TOKEN)
        self.bos_id = self._tok.token_to_id(BOS_TOKEN)
        self.eos_id = self._tok.token_to_id(EOS_TOKEN)
        self.im_start_id = self._tok.token_to_id(IM_START)
        self.im_end_id = self._tok.token_to_id(IM_END)

    # ---------------- 基本信息 ----------------
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def __repr__(self) -> str:
        return f"NovelTokenizer(vocab_size={self.vocab_size}, path={self.path.name})"

    # ---------------- 编码 / 解码 ----------------
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """文本 → id 列表。

        add_special_tokens=False 是预训练的默认值：
        纯文本流进去，模型学的是"下一个字符/词是什么"，
        不需要每句话都插 <|bos|>。
        """
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """id 列表 → 文本。skip_special_tokens=True 会把特殊 token 藏起来。"""
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    # ---------------- 对话模板（SFT 阶段用）----------------
    def apply_chat_template(
        self,
        instruction: str,
        response: str | None = None,
    ) -> str:
        """拼成 SFT 训练用的对话格式。

        产出长这样（和 doupo.jsonl 里的一条对应）：
            <|im_start|>user
            根据标题《第一章 陨落的天才》写一篇玄幻小说章节<|im_end|>
            <|im_start|>assistant
            第一章 陨落的天才
            ...<|im_end|>

        response=None 时只拼到 assistant 的开头 —— 那是推理时喂给模型的提示词。
        """
        parts = [
            f"{IM_START}user\n{instruction}{IM_END}\n",
            f"{IM_START}assistant\n",
        ]
        if response is not None:
            parts.append(f"{response}{IM_END}\n")
        return "".join(parts)

    def encode_chat(self, instruction: str, response: str) -> list[int]:
        """对话 → id，并且**只对 assistant 的回答算 loss**。

        做法是把提示词部分的标签设成 -100（PyTorch 的 ignore_index），
        交叉熵遇到 -100 就跳过 —— 这样模型只学"怎么回答"，
        不学"怎么背题面"。

        返回值:
            input_ids : 送进模型的完整 id 序列
            labels    : 同长度，提示词部分是 -100
        """
        prompt = self.apply_chat_template(instruction, None)
        full = self.apply_chat_template(instruction, response)

        prompt_ids = self.encode(prompt)
        full_ids = self.encode(full)

        # 提示词长度 = 要被 mask 掉的前缀长度
        n_prompt = len(prompt_ids)
        labels = [-100] * n_prompt + full_ids[n_prompt:]
        return full_ids, labels
