"""BF16 normalization must not change when a prefix has different batch peers."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops.kimi_k3 import attention_residual
from atom.model_ops.kimi_k3.activations import rmsnorm_gated
from atom.model_ops.kimi_k3.attention_residual import (
    _apply_attn_res_impl,
    _attn_res_bf16_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


def _tile(x, rows):
    return x.repeat((rows + len(x) - 1) // len(x), *([1] * (x.ndim - 1)))[
        :rows
    ].contiguous()


def _check_cpu_fp32(actual, reference):
    a, b = actual.cpu().float().flatten(1), reference.float().flatten(1)
    difference = a - b
    rms = b.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-10)
    assert (difference.norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-10)).max() < 0.01
    assert (difference.abs() / (0.02 * (b.abs() + rms))).max() <= 1


@pytest.mark.parametrize("heads", [12, 24])
@torch.inference_mode()
def test_bf16_gated_norm_prefill_batch_invariance(heads):
    generator = torch.Generator().manual_seed({12: 20261031, 24: 20261025}[heads])
    # KDA recurrent outputs are small; this scale exposes BF16 rounding near
    # the normalization epsilon at the former one-/multirow crossover.
    x_cpu = (torch.randn(16, heads, 128, generator=generator) * 0.001).bfloat16()
    gate_cpu = torch.randn(16, heads, 128, generator=generator).bfloat16()
    weight_cpu = torch.randn(128, generator=generator)
    reference = (
        x_cpu.float()
        * torch.rsqrt(x_cpu.float().square().mean(-1, keepdim=True) + 1e-5)
        * weight_cpu
        * gate_cpu.float().sigmoid()
    )
    weight = weight_cpu.cuda()
    baseline = None
    for rows in (97, 512, 1792, 3584, 8192):
        x = _tile(x_cpu, rows).cuda()
        storage = torch.empty(
            rows, 4 * heads * 128 + heads + 128, device="cuda", dtype=torch.bfloat16
        )
        gate = storage[:, 3 * heads * 128 : 4 * heads * 128].view(rows, heads, 128)
        gate.copy_(_tile(gate_cpu, rows))
        output = rmsnorm_gated(x, weight, gate, 1e-5, is_prefill=True)
        _check_cpu_fp32(output[:16], reference)
        if baseline is None:
            baseline = output[:16].clone()
        assert torch.equal(output, _tile(baseline, rows))


def _step(monkeypatch, *, one_tile, is_prefill=False):
    monkeypatch.setenv("ATOM_DMLA_ATTN_RES_TILE", "1" if one_tile else "0")
    context = SimpleNamespace(context=SimpleNamespace(is_prefill=is_prefill))
    monkeypatch.setattr(attention_residual, "get_forward_context", lambda: context)


@pytest.mark.parametrize(
    "one_tile,is_prefill,tiles",
    [
        (False, False, [2] * 5),
        (True, True, [2] * 5),
        (True, False, [2, 4, 8, 16, 16]),
    ],
)
def test_bf16_attn_res_tile_follows_the_layer_in_decode(
    one_tile, is_prefill, tiles, monkeypatch
):
    _step(monkeypatch, one_tile=one_tile, is_prefill=is_prefill)
    configs = [_attn_res_bf16_config(c) for c in (2, 3, 5, 9, 17)]
    assert all(c[:2] == (8, 2) for c in configs)
    assert [c[2] for c in configs] == tiles


def test_bf16_attn_res_tile_keeps_two_under_a_column_gather(monkeypatch):
    _step(monkeypatch, one_tile=True)
    configs = [_attn_res_bf16_config(c, gathered=True) for c in (2, 3, 5, 9, 17)]
    assert configs == [(8, 2, 2)] * 5


def test_bf16_attn_res_tile_needs_a_step(monkeypatch):
    monkeypatch.setenv("ATOM_DMLA_ATTN_RES_TILE", "1")
    no_step = SimpleNamespace(context=None)
    monkeypatch.setattr(attention_residual, "get_forward_context", lambda: no_step)
    assert _attn_res_bf16_config(9) == (8, 2, 2)


@pytest.mark.parametrize("one_tile", [False, True])
@pytest.mark.parametrize("blocks,add_count", [(1, 0), (3, 2), (8, 2)])
@torch.inference_mode()
def test_bf16_attn_res_batch_invariance(blocks, add_count, one_tile, monkeypatch):
    _step(monkeypatch, one_tile=one_tile)
    generator = torch.Generator().manual_seed(20261025 + blocks)
    prefix = torch.randn(8, 7168, generator=generator).bfloat16()
    block = torch.randn(8, blocks, 7168, generator=generator).bfloat16()
    score_weight = torch.randn(7168, generator=generator) * 0.01
    norm_weight = torch.randn(7168, generator=generator).bfloat16()
    adds = [torch.randn(8, 7168, generator=generator).bfloat16() for _ in range(2)]
    combined = prefix.float()
    for value in adds[:add_count]:
        combined = combined + value.float()
    candidates = torch.cat((block.float(), combined.unsqueeze(1)), dim=1)
    scores = (candidates * score_weight).sum(-1) * torch.rsqrt(
        candidates.square().mean(-1) + 1e-5
    )
    mixed = (scores.softmax(-1).unsqueeze(-1) * candidates).sum(1)
    reference = (
        mixed
        * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-5)
        * norm_weight.float()
    )
    score_weight, norm_weight = score_weight.cuda(), norm_weight.cuda()
    baseline = None
    for rows in (8, 20, 97, 512, 513, 8192):
        x = _tile(prefix, rows).cuda()
        residuals = _tile(block, rows).cuda()
        add = [_tile(a, rows).cuda() for a in adds] if add_count else [None, None]
        output, stored_prefix = _apply_attn_res_impl(
            x,
            residuals,
            score_weight,
            1e-5,
            add_hidden=add[0],
            out_norm_weight=norm_weight,
            out_eps=1e-5,
            add_hidden2=add[1],
        )
        _check_cpu_fp32(output[:8], reference)
        assert torch.equal(stored_prefix.cpu(), _tile(combined.bfloat16(), rows))
        if baseline is None:
            baseline = output[:8].clone()
        assert torch.equal(output, _tile(baseline, rows))
