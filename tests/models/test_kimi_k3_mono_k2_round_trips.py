# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's opt-in expert-loop switches: the down's ug flags loaded a batch ahead
(``down_peek``) and the ug rounds placed before the queue's last ones
(``ug_place``). Both keep every task's work and result; only when a CTA
learns of it changes."""

import random
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

SWITCHES = {
    "ATOM_MONO_K2_DOWN_PEEK": "down_peek",
    "ATOM_MONO_K2_UG_PLACE": "ug_place",
}


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_default_is_off(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert not getattr(envs, name)
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name)


@pytest.mark.parametrize("name", sorted(SWITCHES))
@pytest.mark.parametrize(
    "requested,votes",
    [(False, (False, True)), (True, (True, False)), (True, (False, False))],
)
def test_every_rank_agrees(monkeypatch, name, requested, votes):
    monkeypatch.setattr(envs, name, requested)
    calls, answers = [], iter(votes)

    def agree(value, group):
        calls.append(value)
        return next(answers)

    monkeypatch.setattr(runner, "group_agree", agree)
    if any(votes):
        assert runner._agree_env(name, None) is votes[0]
    else:
        with pytest.raises(RuntimeError, match=f"{name} differs across TP ranks"):
            runner._agree_env(name, None)
    assert calls == [requested, not requested]


@pytest.mark.parametrize("down_peek", [False, True])
@pytest.mark.parametrize("ug_place", [False, True])
def test_keys_carry_the_switches(down_peek, ug_place):
    obj = object.__new__(runner.K3MonoDecodeRunner)
    obj.skip_padding, obj.debug = False, False
    obj.down_peek, obj.ug_place = down_peek, ug_place
    seam = SimpleNamespace(eps=1e-5, out_eps=1e-5)
    moe_plan = SimpleNamespace(blocks=3, write_idx=-1, seam=seam, ln_eps=1e-5)
    obj.plans = [SimpleNamespace(layer_id=7, moe=moe_plan)]
    key = obj._k2_keys(20, 5)[7]
    assert (key.down_peek, key.ug_place) == (down_peek, ug_place)
    base = k2.K2Build(tokens=20, nblocks=3, qlen=5)
    assert key == replace(base, down_peek=down_peek, ug_place=ug_place)
    sym = symbol_params(key)
    assert (sym["dk"], sym["up"]) == (down_peek, ug_place)
    if down_peek or ug_place:
        assert key_tuple(key, "src") != key_tuple(base, "src")
    assert k2.scratch_bytes(key) == k2.scratch_bytes(base)
    assert k2.plain_bytes(key) == k2.plain_bytes(base)


def _ug_tasks(total, place, seed):
    """``run_ug``'s task sequence on every CTA, the queue's atomics landing in a
    random order across CTAs: {bid: [task, ..]}."""
    n = moe.BLOCKS
    placed = max(total // n - moe.UG_QUEUED, 1) if place else 1
    rng = random.Random(seed)
    queue = 0
    done = {bid: [] for bid in range(n)}
    nxt = {bid: n - 1 - bid for bid in range(n)}
    rounds = {bid: 0 for bid in range(n)}
    live = [bid for bid in range(n) if nxt[bid] < total]
    while live:
        bid = rng.choice(live)
        done[bid].append(nxt[bid])
        rounds[bid] += 1
        r = rounds[bid]
        if r < placed:
            nxt[bid] = n - 1 - bid + n * r
        elif total > n * placed:
            nxt[bid] = n * placed + queue
            queue += 1
        else:
            nxt[bid] = total
        if nxt[bid] >= total:
            live.remove(bid)
    return done, queue


@pytest.mark.parametrize("place", [False, True])
@pytest.mark.parametrize("nu", [1, 16, 21, 22, 43, 62, 86, 113, 188, 270, 350, 448])
def test_ug_runs_every_task_once(place, nu):
    total = nu * moe.UG_GROUPS
    n = moe.BLOCKS
    placed = max(total // n - moe.UG_QUEUED, 1) if place else 1
    for seed in range(3):
        done, _ = _ug_tasks(total, place, seed)
        tasks = sorted(t for ts in done.values() for t in ts)
        assert tasks == list(range(total))
        for bid, ts in done.items():
            rounds = [n - 1 - bid + n * r for r in range(placed)]
            assert ts[:placed] == [t for t in rounds if t < total]


@pytest.mark.parametrize("nu", [62, 113, 188, 350])
def test_ug_place_takes_fewer_claims(nu):
    total = nu * moe.UG_GROUPS
    _, queued = _ug_tasks(total, False, 0)
    _, placed = _ug_tasks(total, True, 0)
    assert placed <= queued
    if total >= (moe.UG_QUEUED + 2) * moe.BLOCKS:
        assert queued - placed == moe.BLOCKS * (total // moe.BLOCKS - moe.UG_QUEUED - 1)


@pytest.mark.parametrize("rows", [5, 20, 60])
def test_compile_keeps_hand_offs_and_resources(rows, tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    # On a GPU host importing aiter already initializes CUDA (Triton's arch probe).
    cuda_was_initialized = torch.cuda.is_initialized()
    seen = {}
    for flags in ((False, False), (True, True)):
        key = k2.K2Build(
            tokens=rows, nblocks=4, qlen=5, down_peek=flags[0], ug_place=flags[1]
        )
        launcher = k2.build(key)
        rec = trace(launcher, k2.ABI)
        assert not rec.unnamed
        access = {(a.stage, a.region, a.space, a.kind) for a in rec.accesses}
        assert ("k2.down", "ugf", Space.SCRATCH, "poll") in access
        assert ("k2.ug", "ugf", Space.SCRATCH, "put") in access
        for region in ("latr", "latg", "fin", "fing"):
            assert any(a[1:] == (region, Space.PEER, "put") for a in access)
        (artifact,) = launcher._mem_cache.values()
        r = kernel_resources(_code_object(artifact._ir_text))
        assert r["scratch"] == r["vgpr_spill"] == 0, r
        assert r["lds"] <= 163840, r
        seen[flags] = (access, r)
    (off, r_off), (on, r_on) = seen[(False, False)], seen[(True, True)]
    assert on == off
    assert r_on["lds"] == r_off["lds"]
    assert cuda_was_initialized or not torch.cuda.is_initialized()
    print({"rows": rows, "off": r_off, "on": r_on})
