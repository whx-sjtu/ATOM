# SPDX-License-Identifier: MIT
"""A sparse MTP verify step plans the per-token `sparse_mtp_` work its sparse
layers read; the dense plan next to it is read only by a dense MLA layer
decoding under the target's metadata. With ATOM_MLA_SKIP_UNREAD_DENSE_PLAN
the builder skips that plan when no such layer exists."""

from types import SimpleNamespace

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


def _dspark():
    return SimpleNamespace(is_block_drafter=True, model=_model(False))


def _runner(model, drafter=None):
    runner = SimpleNamespace(
        model=model, config=SimpleNamespace(speculative_config=object())
    )
    if drafter is not None:
        runner.drafter = drafter
    return runner


def _finish(monkeypatch, runner, is_dummy_run=False):
    monkeypatch.setattr(
        backends.CommonAttentionBuilder, "finish_decode", lambda *a: None
    )
    monkeypatch.setattr(aiter_mla, "kv_indices_generate_triton", lambda *a: None)
    builder = AiterMLAMetadataBuilder.__new__(AiterMLAMetadataBuilder)
    builder.model_runner = runner
    builder.is_sparse = True
    builder.block_ratio = 1
    built = []
    builder._attach_selection = lambda md: None
    builder.set_mla_persistent_worker_buffers = lambda bs, q: (
        built.append(("dense", bs, q))
        or {"work_indptr": "dense", "sparse_dcp_decode_plan": None}
    )
    builder._set_mla_persistent_worker_buffers_sparse_mtp = lambda tokens: (
        built.append(("sparse", tokens)) or {"sparse_mtp_work_indptr": "sparse"}
    )
    builder._build_token_block_tables = lambda *a: None
    builder._decode_index_rows = lambda *a: None
    builder._mtp_index_rows = lambda rows: None
    batch = SimpleNamespace(
        total_seqs_num_decode=3,
        total_tokens_num_decode=15,
        total_tokens_num=15,
        is_dummy_run=is_dummy_run,
    )
    md = SimpleNamespace(
        block_tables=None,
        kv_indices=None,
        kv_indptr=None,
        max_seqlen_k=0,
        work_indptr=None,
        sparse_dcp_decode_plan="stale",
    )
    builder.finish_decode(batch, md, 4, 20, 5)
    return built, md


def test_on_by_default_kimi_k3_dspark_skips_it(monkeypatch):
    monkeypatch.delenv("ATOM_MLA_SKIP_UNREAD_DENSE_PLAN", raising=False)
    built, md = _finish(monkeypatch, _runner(_model(True, True), _dspark()))
    assert built == [("sparse", 15)]
    assert md.work_indptr is None


@pytest.mark.parametrize("flag", ["", "0"])
def test_switched_off_the_dense_plan_is_built(monkeypatch, flag):
    monkeypatch.setenv("ATOM_MLA_SKIP_UNREAD_DENSE_PLAN", flag)
    built, md = _finish(monkeypatch, _runner(_model(True, True), _dspark()))
    assert built == [("dense", 4, 5), ("sparse", 15)]
    assert md.work_indptr == "dense"


def test_kimi_k3_dspark_skips_it(monkeypatch):
    """K3.1: every target MLA layer is sparse; DSpark plans its own block."""
    monkeypatch.setenv("ATOM_MLA_SKIP_UNREAD_DENSE_PLAN", "1")
    built, md = _finish(monkeypatch, _runner(_model(True, True), _dspark()))
    assert built == [("sparse", 15)]
    assert md.work_indptr is None
    assert md.sparse_dcp_decode_plan is None
    assert md.sparse_mtp_work_indptr == "sparse"


def test_a_dummy_run_still_builds_it(monkeypatch):
    monkeypatch.setenv("ATOM_MLA_SKIP_UNREAD_DENSE_PLAN", "1")
    runner = _runner(_model(True, True), _dspark())
    built, _ = _finish(monkeypatch, runner, is_dummy_run=True)
    assert built == [("dense", 4, 5), ("sparse", 15)]


@pytest.mark.parametrize(
    "runner",
    [
        pytest.param(_runner(_model(True, False), _dspark()), id="dense target layer"),
        pytest.param(
            _runner(
                _model(True),
                SimpleNamespace(is_block_drafter=False, model=_model(False)),
            ),
            id="dense non-block drafter",
        ),
        pytest.param(_runner(torch.nn.Linear(2, 2), _dspark()), id="no MLA layer"),
    ],
)
def test_a_dense_reader_keeps_it(monkeypatch, runner):
    monkeypatch.setenv("ATOM_MLA_SKIP_UNREAD_DENSE_PLAN", "1")
    built, md = _finish(monkeypatch, runner)
    assert built == [("dense", 4, 5), ("sparse", 15)]
    assert md.work_indptr == "dense"
