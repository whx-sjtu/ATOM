# SPDX-License-Identifier: MIT
import math

import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

from atom.model_ops import attention_mla

# Kimi-K3.1 DSA decode under QREP + PBM: the DCP group's 96 heads, 512 latent,
# 128 V dims, one row per query token.
HEADS, LORA, VDIM = 96, 512, 128


@pytest.mark.parametrize("rows", [1, 20, 35, 60, 96, 128, 129])
def test_decode_tile_matches_aiters_choice_bit_for_bit(rows, monkeypatch):
    g = torch.Generator(device="cuda").manual_seed(rows)
    o = torch.randn(rows, HEADS, LORA, device="cuda", generator=g)
    x = o.to(torch.bfloat16).transpose(0, 1)
    w_v = 0.05 * torch.randn(HEADS, VDIM, LORA, device="cuda", generator=g)
    w_v = w_v.to(torch.bfloat16)
    outs = []
    for tile in (False, True):
        monkeypatch.setenv("ATOM_DMLA_WV_BMM_TILE", "1" if tile else "0")
        config = attention_mla._v_up_bf16_bmm_config(x)
        assert (config is not None) == (tile and rows <= 128)
        out = torch.full(
            (rows, HEADS, VDIM), math.nan, device="cuda", dtype=torch.bfloat16
        )
        attention_mla._aiter_triton_bf16_bmm(
            x, w_v, YQ=out.transpose(0, 1), config=config
        )
        outs.append(out)
    assert torch.isfinite(outs[1]).all()
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
