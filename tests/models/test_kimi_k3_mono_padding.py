# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Global request padding reaches K2 without a new mask or changed graph width."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from atom.models.kimi_k3.mono import runner
from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.models.kimi_k3.mono.kernels.stages import moe
from atom.models.kimi_k3.mono.plan import state_slots
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import _code_object, kernel_resources, trace
from atom.utils import envs


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("ATOM_MONO_K2_SKIP_PADDING", raising=False)
    assert not envs.ATOM_MONO_K2_SKIP_PADDING


@pytest.mark.parametrize(
    "requested,votes",
    [
        (False, (False, True)),
        (True, (True, False)),
        (True, (False, False)),
        (False, (False, False)),
    ],
)
def test_uniform_startup_arm(monkeypatch, requested, votes):
    monkeypatch.setattr(envs, "ATOM_MONO_K2_SKIP_PADDING", requested)
    calls, answers = [], iter(votes)
    group = object()

    def agree(value, actual):
        assert actual is group
        calls.append(value)
        return next(answers)

    monkeypatch.setattr(runner, "group_agree", agree)
    if any(votes):
        assert runner._agree_skip_padding(group) is votes[0]
    else:
        with pytest.raises(RuntimeError, match="differs across TP ranks"):
            runner._agree_skip_padding(group)
    assert calls == [requested, not requested]


def test_disagreement_precedes_binding(monkeypatch):
    monkeypatch.setattr(runner, "get_tp_group", lambda: SimpleNamespace(cpu_group=None))
    monkeypatch.setattr(runner, "group_agree", lambda *_: False)
    monkeypatch.setattr(
        runner, "bind_agreed", lambda *_: pytest.fail("bind before agreement")
    )
    with pytest.raises(RuntimeError, match="differs across TP ranks"):
        runner.K3MonoDecodeRunner(None)


@pytest.mark.parametrize("qlen", [1, 4])
@pytest.mark.parametrize("active", [0, 36, 40, 44, 48, 64])
def test_request_first_slot_keeps_zero_and_ignores_other_columns(
    monkeypatch, qlen, active
):
    requests = 64 // qlen
    storage = torch.full((requests, qlen + 3), -1, dtype=torch.int32)
    storage[: active // qlen, :qlen] = 0
    storage[: active // qlen, 0] = torch.arange(active // qlen, dtype=torch.int32)
    # Live ReplaySSM requests may leave all non-first columns zero.
    reads = []

    def load(pointer, offset):
        reads.append(offset)
        return int(pointer[offset])

    monkeypatch.setattr(moe, "ld_i32", load)
    c = {
        "skip_padding": True,
        "qlen": qlen,
        "args": {"st_idx": storage.reshape(-1), "st_stride": storage.stride(0)},
    }
    assert [moe.token_live(c, t) for t in range(64)] == [t < active for t in range(64)]
    assert reads == [(t // qlen) * storage.stride(0) for t in range(64)]
    if active in (36, 44):
        # A down-fold wave spans eight tokens, not one uniform request.
        wave = [
            moe.token_live(c, lane // 8)
            for lane in range((active // 8) * 64, (active // 8 + 1) * 64)
        ]
        assert set(wave) == {False, True}


def test_off_never_reads_slot_pointer(monkeypatch):
    monkeypatch.setattr(moe.fx, "Boolean", bool)
    monkeypatch.setattr(moe, "ld_i32", lambda *_: pytest.fail("off read a slot"))
    assert moe.token_live({"skip_padding": False}, 63)
    assert moe.token_live({}, 63)


@pytest.mark.parametrize("rows,qlen", [(64, 1), (64, 4), (32, 4), (64, 2)])
def test_supported_keys_and_cache_identity(rows, qlen):
    base = k2.K2Build(tokens=rows, nblocks=1, qlen=qlen)
    enabled = replace(base, skip_padding=True)
    assert not k2.padding_supported(base)
    assert k2.padding_supported(enabled) is (rows == 64 and qlen in (1, 4))
    assert key_tuple(base, "same source") != key_tuple(enabled, "same source")
    assert symbol_params(enabled)["q"] == qlen
    assert symbol_params(enabled)["sp"] is True
    assert k2.scratch_bytes(base) == k2.scratch_bytes(enabled)
    assert k2.plain_bytes(base) == k2.plain_bytes(enabled)


@pytest.mark.parametrize("qlen", [1, 4])
def test_existing_state_slots_are_views(qlen):
    storage = torch.full((16, 8), -1, dtype=torch.int32)
    storage[:9, :4] = 0
    storage[:9, 0] = torch.arange(9, dtype=torch.int32)
    if qlen == 4:
        source = storage[:, :4]
        md = SimpleNamespace(
            spec_sequence_masks=object(),
            spec_state_indices_tensor=source,
            num_accepted_tokens=torch.full((16,), 4, dtype=torch.int32),
        )
    else:
        source = storage[:, 0].contiguous()
        md = SimpleNamespace(
            spec_sequence_masks=None, non_spec_state_indices_tensor=source
        )
    slots, _ = state_slots(md, torch.ones(16, dtype=torch.int32))
    assert slots.data_ptr() == source.data_ptr()
    assert slots.stride(0) == source.stride(0)
    assert slots[0, 0] == 0 and slots[9, 0] == -1


def _packed_runner():
    obj = object.__new__(runner.K3MonoDecodeRunner)
    scalar = torch.zeros(1)
    attrs = [
        "w_o",
        "w_gate",
        "bias",
        "w_ld",
        "w_sgu",
        "w_sd",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "ln_w",
        "w_up",
    ]
    weight = SimpleNamespace(
        **dict.fromkeys(attrs, scalar),
        write_idx=-1,
        seam=SimpleNamespace(nw=scalar, qk=scalar, out_nw=scalar),
    )
    obj.ones = torch.ones(64, dtype=torch.int32)
    obj.scratch = torch.zeros(16, dtype=torch.uint8)
    obj.queue = torch.zeros(moe.QUEUE_BYTES, dtype=torch.uint8)
    obj.peers = SimpleNamespace(
        plain=torch.zeros(16, dtype=torch.uint8),
        rank=0,
        addresses=torch.zeros(8, dtype=torch.int64),
    )
    obj.plain = obj.peers.plain
    obj._diag_ptr = lambda: 0
    return obj, SimpleNamespace(layer_id=3, moe=weight)


@pytest.mark.parametrize("qlen,enabled", [(1, True), (4, True), (4, False), (2, True)])
def test_real_k2_packing_uses_shared_metadata_or_no_pointer(monkeypatch, qlen, enabled):
    obj, plan = _packed_runner()
    requests = 64 // qlen
    if qlen == 1:
        source = torch.arange(requests, dtype=torch.int32)
        slots = source.unsqueeze(1)
        md = SimpleNamespace(
            spec_sequence_masks=None, non_spec_state_indices_tensor=source
        )
    else:
        slots = torch.arange(requests * 8, dtype=torch.int32).reshape(requests, 8)[
            :, :qlen
        ]
        md = SimpleNamespace(
            spec_sequence_masks=object(),
            spec_state_indices_tensor=slots,
            num_accepted_tokens=torch.ones(requests, dtype=torch.int32),
        )
    key = k2.K2Build(tokens=64, nblocks=1, qlen=qlen, skip_padding=enabled)
    build = SimpleNamespace(k2_keys={3: key}, k2_launchers={key: object()})
    captured = []
    monkeypatch.setattr(runner, "_run_compiled", lambda *args: captured.append(args))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    prefix = torch.zeros((64, 8), dtype=torch.bfloat16)
    obj._k2(build, plan, prefix, prefix, prefix.unsqueeze(1), md)
    args = dict(zip(k2.ABI.names, captured[0][1:-1]))
    if k2.padding_supported(key):
        assert args["st_idx"] == slots.data_ptr() and args["st_stride"] == slots.stride(
            0
        )
    else:
        assert args["st_idx"] == args["st_stride"] == 0


@pytest.mark.parametrize("qlen", [1, 4])
def test_compile_complete_k2_keeps_peer_completion(qlen, tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    # On a GPU host importing aiter already initializes CUDA (Triton's arch probe).
    cuda_was_initialized = torch.cuda.is_initialized()
    resources = []
    for enabled in (False, True):
        key = k2.K2Build(tokens=64, nblocks=1, qlen=qlen, skip_padding=enabled)
        launcher = k2.build(key)
        rec = trace(launcher, k2.ABI)
        access = {(a.region, a.space, a.kind) for a in rec.accesses}
        for region in ("latr", "latg", "fin", "fing"):
            assert (region, Space.PEER, "put") in access
            assert (region, Space.PEER, "poll") in access
        assert not rec.unnamed
        for artifact in launcher._mem_cache.values():
            r = kernel_resources(_code_object(artifact._ir_text))
            assert r["scratch"] == r["vgpr_spill"] == 0, r
            assert r["lds"] <= 163840, r
            resources.append(r)
    assert cuda_was_initialized or not torch.cuda.is_initialized()
    print({"qlen": qlen, "resources": resources})
