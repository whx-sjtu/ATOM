# SPDX-License-Identifier: MIT
"""A cached prefill run sparse sends no sparse MLA layer down the dense
cached-prefix branch, so the builder skips the metadata only that branch
reads -- unless some layer forwarding under the prefill's metadata is dense."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    from atom.model_ops.attention_mla import MLAAttention
    from atom.model_ops.attentions import aiter_mla, backends
    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
except ImportError as _e:  # pragma: no cover - environment, not the code
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "the MLA builder imports AITER")


def _mla(sparse: bool) -> MLAAttention:
    impl = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(impl)
    impl.is_sparse_mla = sparse
    return impl


def _model(*sparse: bool) -> torch.nn.Module:
    return torch.nn.ModuleList([_mla(s) for s in sparse])


def _builder(runner, dcp_world_size=8, kv_shuffled=True) -> AiterMLAMetadataBuilder:
    builder = AiterMLAMetadataBuilder.__new__(AiterMLAMetadataBuilder)
    builder.model_runner = runner
    builder.dcp_world_size = dcp_world_size
    builder.kv_shuffled = kv_shuffled
    builder.attn_prefill_chunk_size = 16
    builder.block_ratio = 1
    builder._sparse_prefill_staged = {"": None}
    return builder


def _runner(model, drafter=None, speculative=False):
    runner = SimpleNamespace(
        model=model,
        config=SimpleNamespace(speculative_config=object() if speculative else None),
        forward_vars={"block_tables": SimpleNamespace(gpu=torch.zeros(4, 4))},
    )
    if drafter is not None:
        runner.drafter = drafter
    return runner


def _finish(monkeypatch, builder, run_sparse):
    monkeypatch.setattr(backends, "build_batch_ids_device", lambda *a, **k: None)
    monkeypatch.setattr(aiter_mla, "kv_indices_generate_triton", lambda *a: None)
    built = []
    builder._build_mla_chunk_meta_dcp = lambda batch: built.append("dcp") or "meta"
    builder._build_mla_chunk_meta = lambda batch: built.append("chunk") or "meta"
    builder._build_mla_chunk_meta_shuffle = lambda batch, prefix: (
        built.append("chunk") or "meta"
    )
    builder._attach_kv_shuffled_prefill_meta = lambda *a: built.append("csr")
    builder._sparse_prefill_staged = {"": None}
    batch = SimpleNamespace(scheduled_bs=1, cached_lens=np.array([64]))
    attn_metadata = SimpleNamespace(
        has_cached=True,
        run_sparse=run_sparse,
        context_lens=None,
        total_kv=80,
        kv_indices=None,
        kv_indptr=None,
        max_seqlen_k=80,
    )
    builder._finish_prefill_batch(batch, attn_metadata)
    return built, attn_metadata


def _dspark():
    return SimpleNamespace(is_block_drafter=True, model=_model(False))


@pytest.mark.parametrize("dcp_world_size", [1, 8])
def test_a_sparse_cached_prefill_builds_no_dense_prefix_metadata(
    monkeypatch, dcp_world_size
):
    """K3.1: every MLA layer of the target is sparse, and DSpark's dense ones
    run as a decode."""
    runner = _runner(_model(True, True), drafter=_dspark(), speculative=True)
    builder = _builder(runner, dcp_world_size)
    built, attn_metadata = _finish(monkeypatch, builder, run_sparse=True)
    assert built == []
    assert not hasattr(attn_metadata, "mla_chunk_meta")


@pytest.mark.parametrize("dcp_world_size", [1, 8])
def test_a_dense_cached_prefill_still_builds_it(monkeypatch, dcp_world_size):
    runner = _runner(_model(True, True), drafter=_dspark(), speculative=True)
    builder = _builder(runner, dcp_world_size)
    built, attn_metadata = _finish(monkeypatch, builder, run_sparse=False)
    assert built == (["dcp", "csr"] if dcp_world_size > 1 else ["chunk", "csr"])
    assert attn_metadata.mla_chunk_meta == "meta"


@pytest.mark.parametrize(
    "runner",
    [
        pytest.param(_runner(_model(True, False)), id="dense target layer"),
        pytest.param(
            _runner(
                _model(True),
                drafter=SimpleNamespace(is_block_drafter=False, model=_model(False)),
                speculative=True,
            ),
            id="dense drafter run as a prefill",
        ),
        pytest.param(_runner(torch.nn.Linear(2, 2)), id="no MLA layer"),
        pytest.param(SimpleNamespace(forward_vars={}), id="no model"),
    ],
)
def test_a_dense_layer_keeps_it_on_a_sparse_step(monkeypatch, runner):
    runner.forward_vars = {"block_tables": SimpleNamespace(gpu=torch.zeros(4, 4))}
    built, _ = _finish(monkeypatch, _builder(runner), run_sparse=True)
    assert built == ["dcp", "csr"]


def test_the_answer_waits_for_the_drafter():
    """Before a configured drafter is attached its layers are unknown, so
    nothing is cached; once it is, the answer is kept."""
    runner = _runner(_model(True), speculative=True)
    builder = _builder(runner)
    assert builder._has_dense_prefill_layer() is False
    runner.drafter = SimpleNamespace(is_block_drafter=False, model=_model(False))
    assert builder._has_dense_prefill_layer() is True
    runner.drafter = _dspark()
    assert builder._has_dense_prefill_layer() is True
