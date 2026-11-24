# SPDX-License-Identifier: MIT
"""`fp4_stage` against the MFMA branch of `_dcp_stage_indexer_fp4_prefill`,
byte for byte, on metadata from the production builder. CPU: Triton's
interpreter, in a child process (the flag must precede the triton import)."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton", reason="fp4_stage is two @triton.jit kernels")

ROOT = Path(__file__).resolve().parents[1]
_NEW_EMPTY = torch.Tensor.new_empty


def _poisoned_new_empty(self, *size, **kwargs):
    return _NEW_EMPTY(self, *size, **kwargs).fill_(0xA5)


def _namespace():
    from atom.model_ops.indexer import fp4_stage
    from atom.model_ops.indexer.layout import fp4_scale_byte
    from atom.model_ops.sparse_indexer_fp4 import fp4_index_scale_rows

    namespace = {
        "torch": torch,
        "np": np,
        "MFMA": "mfma",
        "PRESHUFFLE": "preshuffle",
        "fp4_scale_byte": fp4_scale_byte,
        "fp4_index_scale_rows": fp4_index_scale_rows,
        "fp4_stage": fp4_stage,
        "upload_numpy": lambda values, device: torch.from_numpy(values.copy()).to(
            device
        ),
    }
    for filename, names in (
        ("attentions/aiter_mla.py", {"_build_dcp_indexer_fp4_prefill_meta"}),
        (
            "indexer/indexer_forward.py",
            {"_dcp_stage_indexer_fp4_prefill", "_dcp_stage_indexer_fp4_mfma_fused"},
        ),
    ):
        path = ROOT / "atom/model_ops" / filename
        functions = [
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        assert {n.name for n in functions} == names
        exec(  # noqa: S102 -- runs production host bodies without GPU imports
            compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    return namespace


def run_cases(gather_dtype: str) -> None:
    """Every case of the reference test's batches, world sizes and
    interleaves: the packed rows and both staged planes must be equal."""
    ns = _namespace()
    fp4_stage = ns["fp4_stage"]
    generator = torch.Generator().manual_seed(7)
    for world in (2, 8):
        for interleave in (1, 16):
            caches = [
                torch.randint(
                    256, (12, 4, 64, 16), dtype=torch.uint8, generator=generator
                )
                for _ in range(world)
            ]
            scales = [
                torch.randint(
                    256, (12, 2, 32, 4), dtype=torch.uint8, generator=generator
                )
                for _ in range(world)
            ]
            builder = SimpleNamespace(
                model_runner=SimpleNamespace(block_size=64),
                device="cpu",
                block_table_cols=4,
                _fp4_stage_rows=True,
            )
            table = np.array(
                [[7, 3, 0, 0], [11, 2, 9, 0], [5, 1, 0, 0]], dtype=np.int32
            )
            for lengths in (
                np.array([139, 17, 0]),
                np.array([0, 0, 0]),
                np.array([65, 1, 34]),
                np.array([64, 128, 63]),
            ):
                meta = SimpleNamespace()
                lpad = -(-lengths // (world * interleave)) * interleave
                cu_pad, g_cu = np.r_[0, lpad.cumsum()], np.r_[0, lengths.cumsum()]
                ns["_build_dcp_indexer_fp4_prefill_meta"](
                    builder, meta, 3, lpad, cu_pad, g_cu, lengths, table, "mfma"
                )
                order = []
                for b, length in enumerate(lengths):
                    for pos in range(length):
                        rank = (pos // interleave) % world
                        local = (
                            pos // (world * interleave)
                        ) * interleave + pos % interleave
                        order.append(rank * int(cu_pad[-1]) + int(cu_pad[b]) + local)
                meta.dcp_indexer_gather_index = torch.tensor(
                    order, dtype=getattr(torch, gather_dtype)
                )
                packed = []
                for cache, scale in zip(caches, scales):
                    want = torch.cat(
                        (
                            cache[
                                meta.dcp_indexer_fp4_read_page,
                                :,
                                meta.dcp_indexer_fp4_read_row,
                                :,
                            ].flatten(1),
                            scale.view(-1)[meta.dcp_indexer_fp4_read_scale_at],
                        ),
                        dim=1,
                    )
                    got = fp4_stage.pack_rows(
                        cache,
                        scale,
                        meta.dcp_indexer_fp4_read_page,
                        meta.dcp_indexer_fp4_read_row,
                        meta.dcp_indexer_fp4_read_scale_at,
                    )
                    assert got.dtype == want.dtype and torch.equal(got, want)
                    packed.append(want)

                def gather(values, dim, packed=packed):
                    assert dim == 0 and torch.equal(values, packed[0])
                    return torch.cat(packed, dim=0)

                ns["get_dcp_group"] = lambda gather=gather: SimpleNamespace(
                    all_gather=gather
                )
                want, want_scale = ns["_dcp_stage_indexer_fp4_prefill"](
                    caches[0], scales[0], meta, "mfma"
                )
                # Poison the allocator's bytes: every staged byte must be written.
                try:
                    torch.Tensor.new_empty = _poisoned_new_empty
                    got, got_scale = ns["_dcp_stage_indexer_fp4_mfma_fused"](
                        caches[0], scales[0], meta
                    )
                finally:
                    torch.Tensor.new_empty = _NEW_EMPTY
                assert got.shape == want.shape and got_scale.shape == want_scale.shape
                assert torch.equal(got, want), (world, interleave, lengths)
                assert torch.equal(got_scale, want_scale), (world, interleave, lengths)


@pytest.mark.parametrize("gather_dtype", ["int32", "int64"])
def test_fused_staging_matches_reference_bytes_on_the_cpu(gather_dtype):
    code = (
        "import importlib.util;"
        f"s = importlib.util.spec_from_file_location('probe', {__file__!r});"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m);"
        f"m.run_cases({gather_dtype!r})"
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


def test_stage_row_of_inverts_the_stage_slots():
    from atom.model_ops.indexer import fp4_stage

    slots = torch.tensor([64, 65, 66, 0, 1, 130], dtype=torch.int32)
    row_of = fp4_stage.stage_row_of(slots, 192)
    want = torch.full((192,), -1, dtype=torch.int32)
    want[slots.long()] = torch.arange(6, dtype=torch.int32)
    assert torch.equal(row_of, want)


def test_supports_only_the_opus_planes():
    from atom.model_ops.indexer import fp4_stage

    cache = torch.zeros(3, 4, 64, 16, dtype=torch.uint8)
    scale = torch.zeros(3, 2, 32, 4, dtype=torch.uint8)
    assert fp4_stage.supports(cache, scale)
    assert not fp4_stage.supports(cache[:, :, :32], scale)
    assert not fp4_stage.supports(cache, scale[:, :1])
    assert not fp4_stage.supports(cache.view(torch.int16), scale)
