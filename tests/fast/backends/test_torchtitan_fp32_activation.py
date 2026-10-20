"""--activation-in-fp32: fp32 SiLU in torchtitan's dense FeedForward and MoE GroupedExperts.

TPU v7x evaluates a bf16 SiLU in bf16 arithmetic, while a CPU bf16 SiLU rounds once from fp32. The comparisons
below emulate the TPU by rounding every step of x * sigmoid(x) to bf16 (`_silu_in_bf16`) for bf16 inputs.
"""

from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cpu_ci

pytest.importorskip("torchtitan")

from torchtitan.models.common import moe  # noqa: E402
from torchtitan.models.common.feed_forward import FeedForward  # noqa: E402
from torchtitan.models.common.linear import Linear  # noqa: E402

from miles.backends.torchtitan_utils import fp32_activation  # noqa: E402

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

DIM, HIDDEN, TOKENS = 128, 256, 64
_cpu_silu = F.silu


def _silu_in_bf16(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        return _cpu_silu(x)
    return x * (1 / (1 + torch.exp(-x)))  # every op rounds to bf16


@pytest.fixture
def restore_torchtitan(monkeypatch):
    monkeypatch.setattr(FeedForward, "forward", FeedForward.forward)
    monkeypatch.setattr(moe, "F", moe.F)


def _feed_forward(scale: float) -> FeedForward:
    torch.manual_seed(0)
    ffn = FeedForward(
        FeedForward.Config(
            w1=Linear.Config(in_features=DIM, out_features=HIDDEN),
            w2=Linear.Config(in_features=HIDDEN, out_features=DIM),
            w3=Linear.Config(in_features=DIM, out_features=HIDDEN),
        )
    )
    with torch.no_grad():
        for linear in (ffn.w1, ffn.w2, ffn.w3):
            linear.weight.normal_(0, DIM**-0.5)
        ffn.w1.weight.mul_(scale)  # gate pre-activations of magnitude ~scale
    return ffn


def _experts(scale: float) -> moe.GroupedExperts:
    torch.manual_seed(0)
    experts = moe.GroupedExperts(moe.GroupedExperts.Config(dim=DIM, hidden_dim=HIDDEN, num_experts=2))
    with torch.no_grad():
        for w in (experts.w1_EFD, experts.w2_EDF, experts.w3_EFD):
            w.normal_(0, DIM**-0.5)
        experts.w1_EFD.mul_(scale)
    return experts


def _inputs(dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(TOKENS, DIM, generator=torch.Generator().manual_seed(1)).to(dtype)


_TOKENS_PER_EXPERT = torch.tensor([TOKENS // 4, TOKENS - TOKENS // 4], dtype=torch.int32)


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref).abs().sum() / ref.abs().sum()).item()


def test_without_install_the_forward_is_torchtitans_own(restore_torchtitan):
    ffn = _feed_forward(100.0).bfloat16()
    x = _inputs()
    assert moe.F is F
    assert torch.equal(ffn(x), ffn.w2(F.silu(ffn.w1(x)) * ffn.w3(x)))


def test_install_runs_silu_in_fp32_and_keeps_bf16_outputs(restore_torchtitan):
    fp32_activation.install()
    ffn = _feed_forward(100.0).bfloat16()
    x = _inputs()
    expected = ffn.w2(F.silu(ffn.w1(x).float()).bfloat16() * ffn.w3(x))
    out = ffn(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, expected)
    assert moe.F.softmax is F.softmax


def _activation_err(module, run) -> float:
    """Relative error of the SwiGLU activation `silu(gate) * up` (the input of the down projection) against the
    same activation computed in fp32 from the bf16 gate and up projections, so the bf16 matmuls do not count."""
    seen = []

    def record(fn):
        def wrapped(*, A, B_t, offs):
            out = fn(A=A, B_t=B_t, offs=offs)
            seen.append((A, out))
            return out

        return wrapped

    if isinstance(module, FeedForward):
        hooks = [
            module.w1.register_forward_hook(lambda m, i, o: seen.append(("gate", o))),
            module.w3.register_forward_hook(lambda m, i, o: seen.append(("up", o))),
            module.w2.register_forward_pre_hook(lambda m, i: seen.append(("act", i[0]))),
        ]
        run()
        for hook in hooks:
            hook.remove()
        gate, up, act = (t for _, t in seen)
    else:  # GroupedExperts: grouped matmuls in call order gate, up, down
        with mock.patch.object(module, "_grouped_mm", record(module._grouped_mm)):
            run()
        (_, gate), (_, up), (act, _) = seen
    ref = _cpu_silu(gate.float()) * up.float()
    return _rel_err(act, ref)


@pytest.mark.parametrize("scale", [50.0, 100.0, 200.0])
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_fp32_silu_is_closer_to_fp32_on_large_activations(restore_torchtitan, kind, scale):
    if kind == "dense":
        module, x = _feed_forward(scale).bfloat16(), _inputs()
        run = lambda: module(x)  # noqa: E731
    else:
        module, x = _experts(scale), _inputs()
        run = lambda: module(x, _TOKENS_PER_EXPERT)  # noqa: E731
    with mock.patch.object(F, "silu", _silu_in_bf16):
        err_off = _activation_err(module, run)
        fp32_activation.install()
        err_on = _activation_err(module, run)
    print(f"{kind} scale {scale}: activation rel err, bf16 silu {err_off:.4g}, fp32 silu {err_on:.4g}")
    assert err_on < err_off


def test_gradients_flow_through_the_fp32_silu(restore_torchtitan):
    ref_ffn, x_ref = _feed_forward(100.0), _inputs(torch.float32).requires_grad_()
    ref_ffn(x_ref).square().mean().backward()

    fp32_activation.install()
    ffn, x = _feed_forward(100.0).bfloat16(), _inputs().requires_grad_()
    ffn(x).float().square().mean().backward()

    pairs = [(x.grad, x_ref.grad)] + [
        (getattr(ffn, name).weight.grad, getattr(ref_ffn, name).weight.grad) for name in ("w1", "w2", "w3")
    ]
    for grad, ref in pairs:
        assert grad is not None and grad.dtype == torch.bfloat16 and torch.isfinite(grad).all()
        assert F.cosine_similarity(grad.float().flatten(), ref.flatten(), dim=0) > 0.99


def test_gradients_flow_through_the_moe_fp32_silu(restore_torchtitan):
    fp32_activation.install()
    experts, x = _experts(100.0), _inputs().requires_grad_()
    experts(x, _TOKENS_PER_EXPERT).float().square().mean().backward()
    for grad in (x.grad, experts.w1_EFD.grad, experts.w2_EDF.grad, experts.w3_EFD.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_the_patched_forwards_compile_in_one_graph(restore_torchtitan):
    fp32_activation.install()
    ffn, experts, x = _feed_forward(100.0).bfloat16(), _experts(100.0), _inputs()
    torch._dynamo.reset()
    compiled_ffn = torch.compile(ffn, fullgraph=True, backend="eager")
    compiled_experts = torch.compile(experts, fullgraph=True, backend="eager")
    assert torch.equal(compiled_ffn(x), ffn(x))
    assert torch.equal(compiled_experts(x, _TOKENS_PER_EXPERT), experts(x, _TOKENS_PER_EXPERT))
