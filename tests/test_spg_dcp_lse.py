# SPDX-License-Identifier: MIT
"""``ATOM_SPG_DCP_LSE``: the head-major DCP merge's send LSE built in one
kernel. It runs on the CPU through Triton's interpreter, in a child process
(the flag must precede the triton import)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton", reason="the LSE pack is a @triton.jit kernel")

ROOT = Path(__file__).resolve().parents[1]


def _reference(lse, counts):
    """`cp_lse_a2a_head_major`'s torch ops."""
    from atom.model_ops.dcp_ops import mask_empty_dcp_lse

    lse = lse.contiguous().to(torch.float32)
    if counts is not None:
        lse = mask_empty_dcp_lse(lse, counts)
    return lse.t().contiguous()


def _lse(b, h, generator):
    lse = torch.randn(b, h, generator=generator) * 8
    if b and h:
        lse.view(-1)[:: max(1, b * h // 7)] = float("-inf")
        lse.view(-1)[1 :: max(2, b * h // 5)] = float("nan")
        lse.view(-1)[2 :: max(3, b * h // 3)] = float("inf")
    return lse


def run_kernel() -> None:
    from atom.model_ops.dcp_ops import dcp_head_major_lse

    generator = torch.Generator().manual_seed(11)
    for b, h in ((0, 12), (1, 12), (70, 12), (129, 96), (33, 5)):
        base = _lse(b, h, generator)
        views = {
            "dense": base,
            # The attention's [H, B] LSE handed over as a [B, H] view.
            "transposed": base.t().contiguous().t(),
            "bf16": base.to(torch.bfloat16),
        }
        counts = torch.randint(0, 3, (b + 4,), generator=generator, dtype=torch.int32)
        for name, lse in views.items():
            for owned in (None, counts[:b]):
                got = dcp_head_major_lse(lse, owned)
                want = _reference(lse, owned)
                assert got.shape == want.shape and got.dtype == torch.float32
                assert got.is_contiguous()
                assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (
                    b,
                    h,
                    name,
                    owned is None,
                )


def test_send_lse_matches_the_torch_ops_bit_for_bit():
    code = (
        "import importlib.util;"
        f"s = importlib.util.spec_from_file_location('probe', {__file__!r});"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m);"
        "m.run_kernel()"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env={**os.environ, "TRITON_INTERPRET": "1", "HIP_VISIBLE_DEVICES": ""},
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.returncode == 0, out.stderr[-3000:]


class _Group:
    """A DCP group of two; the test's all-to-all records what is sent."""

    world_size = 2
    device_group = None


@pytest.mark.parametrize("switch", ["0", "1"])
def test_head_major_merge_sends_the_same_lse(monkeypatch, switch):
    from atom.model_ops import dcp_ops

    monkeypatch.setenv("ATOM_SPG_DCP_LSE", switch)
    sent = []

    def all_to_all_single(recv, send, group=None):
        sent.append(send.clone())
        recv.copy_(send)

    monkeypatch.setattr(torch.distributed, "all_to_all_single", all_to_all_single)
    monkeypatch.setattr(
        dcp_ops,
        "dcp_head_major_lse",
        lambda lse, counts: _reference(lse, counts),
    )
    monkeypatch.setattr(
        dcp_ops,
        "_dcp_a2a_combine_kernel",
        type("K", (), {"__getitem__": lambda self, grid: lambda *a, **k: None})(),
    )
    generator = torch.Generator().manual_seed(2)
    out = torch.randn(4, 3, 8, generator=generator)
    lse = _lse(3, 4, generator)
    counts = torch.tensor([1, 0, 2, 7], dtype=torch.int32)
    dcp_ops.cp_lse_a2a_head_major(out, lse, _Group(), owned_counts=counts)
    assert torch.equal(sent[0], out)
    want = _reference(lse, counts[:3])
    assert torch.equal(sent[1].view(torch.int32), want.view(torch.int32))
