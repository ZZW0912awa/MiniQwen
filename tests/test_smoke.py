# -*- coding: utf-8 -*-
"""环境自检：确认 torch 装对了、GPU 真的能用。

注意：这是"能 import"之外的检查——CPU 版 torch 也能 import，
只有真正跑一次 GPU 矩阵乘才能确认轮子和显卡架构匹配。

用法:
    py -m pytest tests/ -v
"""

import pytest
import torch


def test_torch_version():
    """torch 必须存在，且版本可从 __version__ 读出。"""
    assert torch.__version__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="本机没有可用的 CUDA 设备")
def test_cuda_available():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
    print(f"\n设备: {torch.cuda.get_device_name(0)}")
    print(f"算力: {torch.cuda.get_device_capability(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="本机没有可用的 CUDA 设备")
def test_gpu_matmul():
    """真实做一次矩阵乘法的前向 + 反向。

    如果 torch 是 CPU 版，或轮子的算力架构和显卡不匹配，
    这里会抛 'no kernel image is available for execution on the device'。
    """
    a = torch.randn(512, 512, device="cuda", requires_grad=True)
    b = torch.randn(512, 512, device="cuda")
    c = (a @ b).sum()
    c.backward()

    assert a.grad is not None
    assert torch.isfinite(a.grad).all()
    assert not torch.isnan(c)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="本机没有可用的 CUDA 设备")
def test_bf16_supported():
    """Blackwell 之后的卡都应支持 bf16；训练全程要用它混合精度。"""
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    assert y.dtype == torch.bfloat16
