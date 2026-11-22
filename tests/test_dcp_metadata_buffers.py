# SPDX-License-Identifier: MIT
"""Run the source-bound FP4 producers with real CPU buffers and H2D owners."""

import ast
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from atom.model_ops.indexer.layout import fp4_scale_byte
from atom.model_ops.indexer.workspace import DcpGather, dcp_gather_index
from atom.model_ops.sparse_indexer_fp4 import fp4_index_scale_rows
from atom.utils import CpuGpuBuffer, envs
from atom.utils.h2d import PublicationError

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "atom/model_ops/attentions/aiter_mla.py"
RUNNER = ROOT / "atom/model_engine/model_runner.py"
NAMES = ("dcp_fp4_prefill_gather", "dcp_fp4_prefill_slots")


def source_methods(path, names, scope):
    tree = ast.parse(path.read_text())
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    assert methods.keys() == set(names)
    exec(  # noqa: S102 - unchanged production methods at CPU device boundaries
        compile(
            ast.Module(body=list(methods.values()), type_ignores=[]), str(path), "exec"
        ),
        scope,
    )
    return {name: scope[name] for name in names}


class Completion:
    def __init__(self):
        self.records = self.waits = 0

    def record(self, stream=None):
        self.records += 1

    def synchronize(self):
        self.waits += 1


def make_builder(monkeypatch, *, enabled=True, capacity=257, bind=True, **config):
    monkeypatch.setenv("ATOM_DCP_FP4_PREFILL_PINNED", "1" if enabled else "0")
    monkeypatch.setattr(torch.cuda, "Event", Completion)
    uploads, allocations = [], []

    def allocate(*size, **kwargs):
        allocations.append((size, kwargs.copy()))
        return CpuGpuBuffer(*size, **kwargs, pin_memory=False)

    def upload(array, device):
        uploads.append(array.copy())
        return torch.from_numpy(array.copy()).to(device)

    scope = {
        "np": np,
        "torch": torch,
        "envs": envs,
        "CpuGpuBuffer": allocate,
        "DcpGather": DcpGather,
        "dcp_gather_index": dcp_gather_index,
        "fp4_scale_byte": fp4_scale_byte,
        "fp4_index_scale_rows": fp4_index_scale_rows,
        "upload_numpy": upload,
    }
    methods = source_methods(
        BUILDER,
        (
            "_allocate_dcp_fp4_prefill_buffers",
            "_dcp_fp4_prefill_buffer",
            "_build_dcp_indexer_prefill_meta",
            "_build_dcp_indexer_fp4_prefill_meta",
            "_dcp_shard_lengths",
            "_dcp_gathers",
            "_dcp_pass_gathers",
        ),
        scope,
    )
    builder = type("Builder", (), methods)()
    options = {
        "max_model_len": capacity,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_tbo": False,
        "enable_tbo_decode": False,
    }
    options.update(config)
    runner = NS(config=NS(**options), forward_vars={}, block_size=64, device="cpu")
    builder.model_runner, builder.device = runner, "cpu"
    builder._indexer_fp4 = True
    builder._fp4_prefill_combined = True
    builder.dcp_world_size, builder.cp_kv_cache_interleave_size = 8, 1
    builder.block_table_cols = max(8, (capacity + 63) // 64)
    builder._allocate_dcp_fp4_prefill_buffers()
    if bind:
        # Exercise the actual ring/binding order, replacing only CUDA events.
        for name, group in (("sequence", "seq_metadata"), ("position", "positions")):
            runner.forward_vars[name] = CpuGpuBuffer(
                1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=False,
                publication_group=group,
            )
        runtime = {
            "torch": torch,
            "CpuGpuBuffer": CpuGpuBuffer,
            "logger": logging.getLogger(),
        }
        for method in source_methods(
            RUNNER, ("_init_forward_vars_ring", "_init_h2d_publication"), runtime
        ).values():
            method(runner)
    builder.uploads, builder.allocations = uploads, allocations
    return builder


def prepare(builder, lengths, *, prefix="", cached=False):
    runner = builder.model_runner
    cu = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    runner.forward_vars[prefix + ("cu_seqlens_k" if cached else "cu_seqlens_q")] = NS(
        np=cu
    )
    width = max(builder.block_table_cols, (max(lengths, default=0) + 63) // 64)
    tables = (
        np.arange(len(lengths) * width, dtype=np.int32).reshape(len(lengths), width) + 3
    )
    runner.forward_vars[prefix + "block_tables"] = NS(np=tables)
    return NS(has_cached=cached)


def run(builder, lengths, **kwargs):
    runner = builder.model_runner
    owner = getattr(runner, "h2d_owner", None)
    if owner:
        owner.begin()
        runner.h2d_groups["prefill_metadata"].set_counts({"sequence": 0, "position": 0})
        runner.h2d_groups["prefill_metadata"].publish()
    metadata = prepare(builder, lengths, **kwargs)
    builder._build_dcp_indexer_prefill_meta(
        metadata, len(lengths), kwargs.get("prefix", "")
    )
    if owner:
        owner.finish()
    return metadata


def assert_metadata_equal(left, right):
    assert vars(left).keys() == vars(right).keys()
    for name, value in vars(left).items():
        other = getattr(right, name)
        if isinstance(value, torch.Tensor):
            assert value.dtype == other.dtype
            torch.testing.assert_close(value, other, rtol=0, atol=0)
        else:
            assert value == other


def test_flag_default_and_capacity(monkeypatch):
    monkeypatch.delenv("ATOM_DCP_FP4_PREFILL_PINNED", raising=False)
    assert not envs.ATOM_DCP_FP4_PREFILL_PINNED
    builder = make_builder(monkeypatch, capacity=1048576)
    buffers = [builder.model_runner.forward_vars[name] for name in NAMES]
    assert [buffer.np.size for buffer in buffers] == [1048576, 1179648]
    assert (
        sum(buffer.cpu.numel() * buffer.cpu.element_size() for buffer in buffers)
        == 8912896
    )
    assert (
        sum(buffer.gpu.numel() * buffer.gpu.element_size() for buffer in buffers)
        == 8912896
    )
    assert all(
        kwargs["publication_group"] == name
        for name, (_, kwargs) in zip(NAMES, builder.allocations)
    )
    group = builder.model_runner.h2d_groups["prefill_metadata"]
    assert not set(NAMES) & group.indices.keys()
    assert all(
        builder.model_runner.h2d_groups[name].transport == "direct" for name in NAMES
    )


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": False},
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"enable_tbo": True},
        {"enable_tbo_decode": True},
    ],
)
def test_disabled_and_unsupported_topology_allocate_nothing(monkeypatch, options):
    builder = make_builder(monkeypatch, bind=False, **options)
    assert not builder.allocations
    assert not builder._fp4_prefill_pinned


@pytest.mark.parametrize("combined", [False, True])
def test_long_short_empty_and_boundary_requests_reuse_active_prefix(
    monkeypatch, combined
):
    actual = make_builder(monkeypatch)
    reference = make_builder(monkeypatch, enabled=False)
    actual._fp4_prefill_combined = reference._fp4_prefill_combined = combined
    buffers = [actual.model_runner.forward_vars[name] for name in NAMES]
    pointers = [(buffer.cpu.data_ptr(), buffer.gpu.data_ptr()) for buffer in buffers]
    previous = None
    for length in [257, 1, 0, 7, 8, 9, 63, 64, 65, 257]:
        result = run(actual, [length], cached=bool(length % 2))
        expected = run(reference, [length], cached=bool(length % 2))
        assert_metadata_equal(result, expected)
        assert result.dcp_indexer_gather_index.numel() == length
        counts = (length, length + (length + 7) // 8)
        if previous is not None:
            for buffer, old, count in zip(buffers, previous, counts):
                np.testing.assert_array_equal(buffer.np[count:], old[0][count:])
                torch.testing.assert_close(buffer.gpu[count:], old[1][count:])
        previous = [(buffer.np.copy(), buffer.gpu.clone()) for buffer in buffers]
        assert pointers == [
            (buffer.cpu.data_ptr(), buffer.gpu.data_ptr()) for buffer in buffers
        ]
    assert len(actual.uploads) == 2 * 10  # only local_cu and small staged tables
    assert len(reference.uploads) == 4 * 10
    assert actual.model_runner._stage_h2d_done.waits == 10


@pytest.mark.parametrize(
    "lengths,prefix,bind",
    [
        ([258], "", True),
        ([7, 9], "", True),
        ([9], "ub0_", True),
        ([9], "", False),
    ],
)
def test_fallbacks_preserve_buffers_and_metadata(monkeypatch, lengths, prefix, bind):
    actual = make_builder(monkeypatch, bind=bind)
    reference = make_builder(monkeypatch, enabled=False, bind=bind)
    before = [actual.model_runner.forward_vars[name].np.copy() for name in NAMES]
    assert_metadata_equal(
        run(actual, lengths, prefix=prefix), run(reference, lengths, prefix=prefix)
    )
    assert len(actual.uploads) == 4
    for name, snapshot in zip(NAMES, before):
        np.testing.assert_array_equal(
            actual.model_runner.forward_vars[name].np, snapshot
        )


def test_fp8_multipass_never_borrows_fp4_storage(monkeypatch):
    builder = make_builder(monkeypatch)
    attn = prepare(builder, [9, 17])
    passes = [NS(first=0, end=1), NS(first=1, end=2)]
    gathers = builder._dcp_pass_gathers(attn, 2, "", passes)
    assert [g.gather_index.numel() for g in gathers] == [9, 17]
    assert len(builder.uploads) == 4
    assert all(not builder.model_runner.forward_vars[name].np.any() for name in NAMES)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "state", ["sealed", "dirty", "published", "wrong_owner", "wrong_thread"]
)
def test_protocol_failure_precedes_host_mutation(monkeypatch, name, state):
    builder = make_builder(monkeypatch)
    runner = builder.model_runner
    buffer = runner.forward_vars[name]
    owner = runner.h2d_owner
    owner.begin()
    if state == "sealed":
        owner.finish()
    elif state == "dirty":
        buffer._publication._writable = False
    elif state == "published":
        buffer.copy_to_gpu(0)
    elif state == "wrong_owner":
        runner.h2d_owner = make_builder(monkeypatch).model_runner.h2d_owner
    before = buffer.np.copy()

    def produce():
        selected = builder._dcp_fp4_prefill_buffer(name, 3, 1, "")
        selected.np[:3] = 123
        selected.copy_to_gpu(3)

    with pytest.raises(PublicationError):
        if state == "wrong_thread":
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(produce).result()
        else:
            produce()
    np.testing.assert_array_equal(buffer.np, before)


def test_current_owner_lookup_does_not_cache_initial_buffer(monkeypatch):
    builder = make_builder(monkeypatch)
    replacement = make_builder(monkeypatch)
    first = builder.model_runner.forward_vars
    builder.model_runner.forward_vars = replacement.model_runner.forward_vars
    builder.model_runner.h2d_owner = replacement.model_runner.h2d_owner
    builder.model_runner.h2d_groups = replacement.model_runner.h2d_groups
    run(builder, [17])
    assert first[NAMES[0]].np.sum() == 0
    assert replacement.model_runner.forward_vars[NAMES[0]].np[:17].sum() > 0
