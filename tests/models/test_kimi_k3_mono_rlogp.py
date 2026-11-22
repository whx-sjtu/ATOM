# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2 router-partial addressing, initialization and compile-only handoffs."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from atom.models.kimi_k3.mono import runner
from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.models.kimi_k3.mono.kernels.stages import moe
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import _code_object, kernel_resources, trace
from atom.utils import envs


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("ATOM_MONO_K2_RLOGP", raising=False)
    assert not envs.ATOM_MONO_K2_RLOGP


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("votes", [(True, False), (False, True), (False, False)])
def test_ranks_agree_before_peer_allocation(monkeypatch, requested, votes):
    monkeypatch.setattr(envs, "ATOM_MONO_K2_RLOGP", requested)
    calls = []
    answers = iter(votes)
    group = object()

    def agree(value, actual_group):
        assert actual_group is group
        calls.append(value)
        return next(answers)

    monkeypatch.setattr(runner, "group_agree", agree)
    if any(votes):
        assert runner._agree_rlogp(group) is votes[0]
    else:
        with pytest.raises(RuntimeError, match="differs across TP ranks"):
            runner._agree_rlogp(group)
    assert calls == [requested, not requested]


def test_mismatched_arm_never_binds_or_allocates(monkeypatch):
    monkeypatch.setattr(runner, "get_tp_group", lambda: SimpleNamespace(cpu_group=None))
    monkeypatch.setattr(runner, "group_agree", lambda *_: False)
    monkeypatch.setattr(
        runner, "bind_agreed", lambda *_: pytest.fail("binding before agreement")
    )
    monkeypatch.setattr(
        runner,
        "owned_peer_buffer",
        lambda *_: pytest.fail("allocation before agreement"),
    )
    with pytest.raises(RuntimeError, match="differs across TP ranks"):
        runner.K3MonoDecodeRunner(None)


@pytest.mark.parametrize("rows", [4, 8, 12, 16, 48, 64])
def test_all_shallow_widths_have_the_actual_lds_guard(rows):
    minimum_blocks = {4: 6, 8: 3, 12: 2, 16: 1, 48: 1, 64: 1}[rows]
    for blocks in range(9):
        key = k2.K2Build(tokens=rows, nblocks=blocks, rlog_parts=True)
        assert k2.rlogp_supported(key) is (blocks >= minimum_blocks)
        assert not k2.rlogp_supported(replace(key, rlog_parts=False))


@pytest.mark.parametrize("rows", [48, 64])
def test_global_rows_map_all_parts_without_group_aliases(rows):
    written = set()
    for rank in range(8):
        for part in range(4):
            for first in range(0, rows, 16):
                for local in range(min(16, rows - first)):
                    for column in range(112):
                        written.add(
                            moe.router_part_index(
                                rows, rank, part, first + local, column
                            )
                        )
    assert written == set(range(rows * 896 * 4))
    for token in range(rows):
        for expert in range(896):
            for part in range(4):
                at = moe.router_part_index(
                    rows, expert // 112, part, token, expert % 112
                )
                assert at in written
    routes = [
        bid + 56 * i for bid in range(56) for i in range(2) if bid + 56 * i < rows
    ]
    assert sorted(routes) == list(range(rows))


@pytest.mark.parametrize("rows", [48, 64])
def test_optional_region_preserves_the_native_layout_and_indexer_boundary(rows):
    native = k2.peer_layout(rows)
    enabled = k2.peer_layout(rows, True)
    assert "rlogp" not in native
    assert {name: enabled[name] for name in native} == native
    offset, size = enabled["rlogp"]
    assert offset >= k2.peer_bytes(rows)
    assert size == rows * 896 * 4 * 8
    assert k2.peer_bytes(rows, True) - k2.peer_bytes(rows) == size
    for width in range(1, rows + 1):
        assert runner.ix_peer_offset(rows, True) >= k2.peer_bytes(width, True)
    assert runner.ix_peer_offset(rows) < runner.ix_peer_offset(rows, True)


def test_runner_binds_the_agreed_arm_into_cache_and_symbol_identity():
    obj = object.__new__(runner.K3MonoDecodeRunner)
    obj.debug = False
    obj.plans = [
        SimpleNamespace(
            layer_id=17,
            moe=SimpleNamespace(
                blocks=2,
                write_idx=-1,
                seam=SimpleNamespace(eps=1e-5, out_eps=1e-5),
                ln_eps=1e-5,
            ),
        )
    ]
    obj.rlog_parts = False
    native = obj._k2_keys(64)[17]
    obj.rlog_parts = True
    enabled = obj._k2_keys(64)[17]
    assert replace(native, rlog_parts=True) == enabled
    assert key_tuple(native, "same sources") != key_tuple(enabled, "same sources")
    assert symbol_params(native)["rp"] is False
    assert symbol_params(enabled)["rp"] is True


@pytest.mark.parametrize("rows,blocks", [(48, 1), (64, 1), (64, 8), (4, 1)])
def test_compile_only_uses_the_selected_mailbox_and_records_resources(
    tmp_path, monkeypatch, rows, blocks
):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    resources = []
    for enabled in (False, True):
        key = k2.K2Build(tokens=rows, nblocks=blocks, rlog_parts=enabled)
        launcher = k2.build(key)
        record = trace(launcher, k2.ABI)
        selected = "rlogp" if k2.rlogp_supported(key) else "rlog"
        other = "rlog" if selected == "rlogp" else "rlogp"
        accesses = {(a.region, a.space, a.kind) for a in record.accesses}
        assert (selected, Space.PEER, "put") in accesses
        assert (selected, Space.PEER, "poll") in accesses
        assert not any(a.region == other for a in record.accesses)
        assert not record.unnamed
        for artifact in launcher._mem_cache.values():
            resource = kernel_resources(_code_object(artifact._ir_text))
            assert resource["scratch"] == 0, resource
            assert resource["vgpr_spill"] == 0, resource
            # SGPR-to-VGPR moves also occur in the native build. Preserve the
            # count for matched resource review; scratch/VGPR spills are fatal.
            assert isinstance(resource["sgpr_spill"], int), resource
            resources.append(resource)
    assert resources
    assert not torch.cuda.is_initialized()
    print({"rows": rows, "blocks": blocks, "resources": resources})
