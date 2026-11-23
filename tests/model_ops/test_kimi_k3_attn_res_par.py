# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A row-tail step's zero addends are read as one zero row: the same kernel,
bit for bit."""

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops.kimi_k3 import attention_residual
from atom.model_ops.kimi_k3.attention_residual import (
    _apply_attn_res_impl,
    zero_addends,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)

HIDDEN = 7168
BANK = 8
SWITCH = "ATOM_KIMI_K3_ATTN_RES_SKIP_ZERO_ADDENDS"


def _inputs(tokens, seed):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=generator) * scale).bfloat16().cuda()

    return {
        "bank": torch.cat(
            [randn(tokens, 1, HIDDEN, scale=1 + b) for b in range(BANK)], 1
        ),
        "prefix": randn(tokens, HIDDEN, scale=4.0),
        "adds": [randn(tokens, HIDDEN) for _ in range(2)],
        "score_weight": randn(HIDDEN, scale=0.03).float(),
        "out_norm": randn(HIDDEN),
    }


def _assert_bitwise(got, want, what=""):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape, what
        assert torch.equal(g.view(torch.uint8), w.view(torch.uint8)), what


@pytest.fixture
def switch(monkeypatch):
    attention_residual._ZERO_ADDENDS.clear()
    yield lambda on: monkeypatch.setenv(SWITCH, "1" if on else "0")
    attention_residual._ZERO_ADDENDS.clear()


def _mix(x, blocks, adds):
    return _apply_attn_res_impl(
        x["prefix"],
        x["bank"],
        x["score_weight"],
        1e-5,
        adds[0],
        x["out_norm"],
        1e-5,
        adds[1],
        num_blocks=blocks,
    )


def _compiled_kernels():
    caches = attention_residual._attn_res_fused_kernel.device_caches.values()
    return sum(len(c[0]) for c in caches)


@pytest.mark.parametrize("tokens", [1, 7, 130, 1029])
@pytest.mark.parametrize("on", [False, True])
@torch.inference_mode()
def test_zero_pair_matches_loading_zeros(switch, tokens, on):
    switch(on)
    x = _inputs(tokens, 20261123 + tokens)
    for blocks in range(1, BANK + 1):
        want = _mix(x, blocks, [torch.zeros_like(x["prefix"])] * 2)
        compiled = _compiled_kernels()
        pair = zero_addends(x["prefix"])
        assert not pair[0].any() and not pair[1].any()
        assert len(attention_residual._ZERO_ADDENDS) == (2 if on else 0)
        got = _mix(x, blocks, pair)
        _assert_bitwise(got, want, f"{blocks} blocks")
        assert _compiled_kernels() == compiled, "the zero row took another kernel"
        assert not attention_residual._ZERO_ADDENDS
        # prefix_out is a fresh tensor either way
        assert got[1].data_ptr() != x["prefix"].data_ptr()
    if on:  # one row per device, dtype and width, never written
        assert len(attention_residual._ZERO_ROWS) == 1
    assert not any(row.any() for row in attention_residual._ZERO_ROWS.values())


@torch.inference_mode()
def test_the_custom_op_sees_the_pair_itself(switch):
    switch(True)
    x = _inputs(16, 20261127)
    want = _mix(x, 5, [torch.zeros_like(x["prefix"])] * 2)
    pair = zero_addends(x["prefix"])
    got = attention_residual.apply_attn_res(
        x["prefix"],
        x["bank"],
        x["score_weight"],
        1e-5,
        pair[0],
        x["out_norm"],
        1e-5,
        pair[1],
        num_blocks=5,
    )
    assert not attention_residual._ZERO_ADDENDS
    _assert_bitwise(got, want)


@pytest.mark.parametrize("handed", ["copies", "one_real", "swapped", "none"])
@torch.inference_mode()
def test_only_the_registered_pair_skips(switch, handed):
    switch(True)
    x = _inputs(16, 20261124)
    zeros = torch.zeros_like(x["prefix"])
    real = x["adds"][0]
    want = _mix(
        x,
        3,
        {
            "copies": [zeros, zeros],
            "one_real": [real, zeros],
            "swapped": [zeros, zeros],
            "none": [None, None],
        }[handed],
    )
    pair = zero_addends(x["prefix"])
    adds = {
        "copies": [p.clone() for p in pair],
        "one_real": [real, pair[1]],
        "swapped": [pair[1], pair[0]],
        "none": [None, None],
    }[handed]
    _assert_bitwise(_mix(x, 3, adds), want)
    # the next mix ends the registration, hit or not
    assert not attention_residual._ZERO_ADDENDS
    # so a later mix handed the pair loads it
    _assert_bitwise(_mix(x, 3, pair), _mix(x, 3, [zeros, zeros]))
