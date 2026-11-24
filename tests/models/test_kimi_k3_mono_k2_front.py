# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's opt-in front switches (``layer.FRONT``): the router's and latent's x
staged further ahead (``early_ahead``), two latent passes a poll
(``latq_batch``), the shared gate/up's x staged after theirs (``early_yield``).
Each keeps every sum and its order; only when loads and polls are issued
changes."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from atom.models.kimi_k3.mono import runner
from atom.models.kimi_k3.mono.config import MAX_ROWS, groups
from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.models.kimi_k3.mono.kernels.stages import moe
from atom.mono.device.sync import POLL_MAX
from atom.mono.plan.execution import THREADS
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import _code_object, kernel_resources, trace
from atom.utils import envs

SWITCHES = dict(runner.K2_FRONT_ENVS)
SYMS = {"early_ahead": "ea", "latq_batch": "lb", "early_yield": "ey"}


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_default_is_off(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert not getattr(envs, name)
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name)


def test_the_switches_are_build_fields():
    assert tuple(SWITCHES.values()) == k2.FRONT
    assert not set(k2.FRONT) & set(k2.HANDOFFS)
    by_name = {f.name: f for f in fields(k2.K2Build)}
    for name in k2.FRONT:
        assert not by_name[name].default
        assert by_name[name].metadata["sym"] == SYMS[name]


def _runner(front):
    obj = object.__new__(runner.K3MonoDecodeRunner)
    obj.skip_padding = obj.debug = obj.down_peek = obj.ug_place = False
    obj.front = front
    seam = SimpleNamespace(eps=1e-5, out_eps=1e-5)
    moe_plan = SimpleNamespace(blocks=3, write_idx=-1, seam=seam, ln_eps=1e-5)
    obj.plans = [SimpleNamespace(layer_id=7, moe=moe_plan)]
    return obj


@pytest.mark.parametrize("rows", [1, 4, 5, 15, 16, 17, 20, 35, 60, 64])
def test_keys_take_a_switch_where_it_changes_the_code(rows):
    on = {name: True for name in k2.FRONT}
    key = _runner(on)._k2_keys(rows, 5)[7]
    assert key.early_ahead == (k2.EARLY_AHEAD if groups(rows) > 1 else 0)
    assert key.latq_batch == (moe.latq_passes(rows) > 1)
    assert key.early_yield
    off = _runner({name: False for name in k2.FRONT})._k2_keys(rows, 5)[7]
    assert off == k2.K2Build(tokens=rows, nblocks=3, qlen=5)
    assert k2.symbol(off) == k2.symbol(k2.K2Build(tokens=rows, nblocks=3, qlen=5))
    sym = k2.symbol(key)
    for name in k2.FRONT:
        v = int(getattr(key, name))
        assert (f"_{SYMS[name]}{v}_" in sym) == bool(v)
        assert f"_{SYMS[name]}0_" not in sym


def test_a_default_build_keeps_its_symbol():
    key = k2.K2Build(tokens=25, nblocks=3)
    assert k2.symbol(key) == "_ZN4atom42k3_mono_k2_s25_nb3_r1_q1_sp0_dk0_up0_g0_t0E"
    sym = k2.symbol(replace(key, early_ahead=3, early_yield=True))
    assert sym == "_ZN4atom50k3_mono_k2_s25_nb3_r1_q1_sp0_dk0_up0_ea3_ey1_g0_t0E"


@pytest.mark.parametrize("s", range(1, MAX_ROWS + 1))
def test_the_yield_flags_sit_past_every_region(s):
    """ESTG is laid out last: every other scratch region keeps its offset, and
    it holds a flag per router and latent task."""
    lay = moe.scratch_layout(moe.MoeBuild(tokens=s))
    names = list(lay)
    assert names[-1] == "estg"
    end = max(o + n for name, (o, n) in lay.items() if name != "estg")
    assert lay["estg"][0] >= end
    assert lay["estg"][1] == (moe.R_TASKS + moe.L_TASKS) * 8
    assert moe.R_TASKS + moe.L_TASKS <= THREADS
    assert "estg" in moe.REGIONS


@pytest.mark.parametrize("s", range(1, MAX_ROWS + 1))
def test_batched_latent_gather_covers_every_block_once(s):
    """``load_latq_batched``: each (token, 32-block) by the thread and pass the
    unbatched gather gives it, a poll's pairs within ``POLL_MAX``."""
    nblk, npass = s * (moe.LAT // 32), moe.latq_passes(s)
    assert 5 * moe.LATQ_BATCH <= POLL_MAX
    seen = []
    for j in range(-(-npass // moe.LATQ_BATCH)):
        n = min(moe.LATQ_BATCH, npass - j * moe.LATQ_BATCH)
        for p in range(n):
            i = j * moe.LATQ_BATCH + p
            seen += [tid + THREADS * i for tid in range(THREADS) if tid + THREADS * i < nblk]
    assert seen == list(range(nblk))


def _resources(launcher):
    (artifact,) = launcher._mem_cache.values()
    return kernel_resources(_code_object(artifact._ir_text))


@pytest.mark.parametrize("rows", [5, 20, 60])
def test_compile_keeps_the_front_and_resources(rows, tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    cuda_was_initialized = torch.cuda.is_initialized()
    base = {"batch_polls": True, "route_parts": True, "ln_sq": True}
    arms = {"off": {}, **{n: {n: True} for n in k2.FRONT}, "all": dict.fromkeys(k2.FRONT, True)}
    seen = {}
    for arm, on in arms.items():
        key = k2.K2Build(tokens=rows, nblocks=4, qlen=5, **base, **k2.front_at(rows, on))
        launcher = k2.build(key)
        rec = trace(launcher, k2.ABI)
        assert not rec.unnamed
        access = {(a.stage, a.region, a.space, a.kind) for a in rec.accesses}
        r = _resources(launcher)
        assert r["scratch"] == r["vgpr_spill"] == 0, (arm, r)
        assert r["lds"] <= 163840, (arm, r)
        seen[arm] = (access, r)
    off, r_off = seen["off"]
    for arm in ("early_ahead", "latq_batch"):
        assert seen[arm][0] == off
    ey, _ = seen["early_yield"]
    assert ey - off == {
        ("k2.early", "estg", Space.SCRATCH, "put"),
        ("k2.early", "estg", Space.SCRATCH, "poll"),
    }
    assert seen["all"][0] == ey
    for arm, (_, r) in seen.items():
        assert r["lds"] == r_off["lds"], arm
    assert cuda_was_initialized or not torch.cuda.is_initialized()
    print({"rows": rows, **{a: (r["vgpr"], r["lds"]) for a, (_, r) in seen.items()}})
