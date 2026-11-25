# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_DSA_PREFILL_TOPK_SAMPLED: a prefill band's ids through `score_topk`,
the sampled selector against the one-block kernel (gfx950, OPUS FP4 logits).
Random FP4 codes make a plane of few distinct values: ties everywhere."""

import pytest
import torch

rows_module = pytest.importorskip("atom.model_ops.indexer.rows", exc_type=ImportError)


def _permute_scales(scale):
    n, h = scale.shape[:2]
    return (
        scale.reshape(n, h // 32, 32, 2, 2)
        .permute(0, 4, 2, 3, 1)
        .reshape(n, 2, 32, h // 16)
        .contiguous()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize(
    "ctx,new,taken",
    [(90112, 512, True), (90112, 256, False), (60000, 512, False)],
)
def test_sampled_prefill_topk_keeps_the_one_block_ids(monkeypatch, ctx, new, taken):
    from aiter.jit.utils.chip_info import get_gfx

    from atom.model_ops.indexer import sampled_select, score
    from atom.model_ops.indexer.query import quantize_query

    if get_gfx() != "gfx950":
        pytest.skip("gfx950 OPUS layout")
    torch.manual_seed(907)
    dev, heads, page, topk = "cuda", 32, 64, 2048
    width = ctx + new
    pages = -(-width // page)
    q = torch.randn(new, heads, 128, device=dev).bfloat16()
    angles = torch.randn(new, 32)
    cos, sin = angles.cos().bfloat16().to(dev), angles.sin().bfloat16().to(dev)
    query = quantize_query(
        q,
        cos,
        sin,
        torch.arange(new, device=dev),
        64,
        fp4=True,
        layout=rows_module.MFMA,
    )
    kv = torch.randint(0, 256, (pages, page, 64), dtype=torch.uint8, device=dev)
    ks = torch.randint(122, 129, (pages, page, 4), dtype=torch.uint8, device=dev)
    cache = rows_module.IndexCache(
        kv.reshape(pages, page, 4, 16).permute(0, 2, 1, 3).contiguous(),
        _permute_scales(ks),
        page,
        rows_module.MFMA,
    )
    table = torch.arange(pages, dtype=torch.int32, device=dev)[None]
    starts = torch.tensor([0, new], dtype=torch.int32, device=dev)
    ends = torch.arange(ctx + 1, width + 1, dtype=torch.int32, device=dev)
    weights = torch.randn(new, heads, dtype=torch.bfloat16, device=dev)

    def select(**switch):
        rows = rows_module.fp4_rows(
            ends,
            width,
            None,
            rows_module.Fp4MqaRaggedMetadata(starts, new, table),
            heads=heads,
            page_size=page,
            layout=rows_module.MFMA,
            bands=((0, new),),
        )
        out = torch.full((new, topk), -7, dtype=torch.int32, device=dev)
        score.score_topk(
            query,
            weights,
            cache,
            rows,
            topk=topk,
            weight_scale=0.125,
            stable=True,
            out=out,
            **switch,
        )
        return out

    calls = []
    real = sampled_select.stable_sampled_topk

    def counted(scores, *args):
        calls.append(tuple(scores.shape))
        return real(scores, *args)

    monkeypatch.setattr(sampled_select, "stable_sampled_topk", counted)
    ref = select()
    assert not calls
    got = select(sampled_topk=True)
    torch.cuda.synchronize()
    assert calls == ([(new, width)] if taken else [])
    assert torch.equal(got, ref)
