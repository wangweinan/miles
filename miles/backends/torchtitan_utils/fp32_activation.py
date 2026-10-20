"""SiLU in fp32 inside torchtitan's SwiGLU blocks (--activation-in-fp32), matching SGLANG_TPU_ACTIVATION_FP32=1.

TPU v7x evaluates a bf16 SiLU in bf16 arithmetic, which makes Qwen3's logits several times noisier than an fp32
forward. Only `silu(gate)` moves to fp32; the matmuls and the `gate * up` product stay in the compute dtype.
"""

import types

import torch
import torch.nn.functional as F
from torchtitan.models.common import moe
from torchtitan.models.common.feed_forward import FeedForward


def silu_fp32(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return F.silu(x.float()).to(x.dtype)


def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(silu_fp32(self.w1(x)) * self.w3(x))


def install() -> None:
    FeedForward.forward = _feed_forward
    # GroupedExperts.forward applies F.silu to its bf16 grouped-matmul output; moe.py gets its own `F` whose
    # silu runs in fp32, so the experts' forward (which differs between torchtitan branches) stays untouched.
    functional = types.ModuleType("miles_fp32_silu_functional")
    functional.__dict__.update({k: v for k, v in vars(F).items() if not k.startswith("__")})
    functional.silu = silu_fp32
    moe.F = functional
