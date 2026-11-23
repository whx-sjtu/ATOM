# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's opt-in hand-off switches (``layer.HANDOFFS``): the latent gathered
before the route is waited for (``latq_first``), every pass's pairs in one
poll (``batch_polls``), the router's K parts added by the route
(``route_parts``), lnorm's sums of squares from the owner (``ln_sq``). Each
keeps every sum and its order; only the round trips between them change."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from flydsl.compiler.protocol import dsl_size_of

from atom.models.kimi_k3.mono import runner
from atom.models.kimi_k3.mono.config import MAX_ROWS, groups
from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.models.kimi_k3.mono.kernels.stages import moe
from atom.mono.device import gemv
from atom.mono.plan.execution import THREADS
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import _code_object, kernel_resources, trace
from atom.utils import envs

SWITCHES = dict(runner.K2_HANDOFF_ENVS)
SYMS = {"latq_first": "lf", "batch_polls": "bp", "route_parts": "rp", "ln_sq": "lq"}


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_default_is_on(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name)
    monkeypatch.setenv(name, "0")
    assert not getattr(envs, name)


def test_the_switches_are_build_fields():
    assert tuple(SWITCHES.values()) == k2.HANDOFFS
    by_name = {f.name: f for f in fields(k2.K2Build)}
    for name in k2.HANDOFFS:
        assert by_name[name].default is False
        assert by_name[name].metadata["sym"] == SYMS[name]


def _runner(handoffs):
    obj = object.__new__(runner.K3MonoDecodeRunner)
    obj.skip_padding = obj.debug = obj.down_peek = obj.ug_place = False
    obj.handoffs = handoffs
    seam = SimpleNamespace(eps=1e-5, out_eps=1e-5)
    moe_plan = SimpleNamespace(blocks=3, write_idx=-1, seam=seam, ln_eps=1e-5)
    obj.plans = [SimpleNamespace(layer_id=7, moe=moe_plan)]
    return obj


@pytest.mark.parametrize("on", [(), *[(n,) for n in k2.HANDOFFS], k2.HANDOFFS])
def test_keys_carry_the_switches(on):
    flags = {name: name in on for name in k2.HANDOFFS}
    obj = _runner(flags)
    key = obj._k2_keys(k2.LATQ_FIRST_ROWS, 5)[7]
    base = k2.K2Build(tokens=k2.LATQ_FIRST_ROWS, nblocks=3, qlen=5)
    assert key == replace(base, **flags)
    assert k2.scratch_bytes(key) == k2.scratch_bytes(base)
    assert k2.plain_bytes(key) == k2.plain_bytes(base)
    sym = k2.symbol(key)
    for name in k2.HANDOFFS:
        assert (f"_{SYMS[name]}1_" in sym) == (name in on)
        assert f"_{SYMS[name]}0_" not in sym
    want = tuple(r for r, f in (("rlogp", "route_parts"), ("lsq", "ln_sq")) if f in on)
    assert obj.k2_peer_extra == want == k2.peer_extra(key)


@pytest.mark.parametrize("rows", [5, 10, 15, 20, 60])
def test_the_latent_goes_first_only_on_narrow_widths(rows):
    on = {name: True for name in k2.HANDOFFS}
    key = _runner(on)._k2_keys(rows, 5)[7]
    assert key.latq_first == (rows <= k2.LATQ_FIRST_ROWS)
    assert key.batch_polls and key.route_parts and key.ln_sq
    off = _runner({name: False for name in k2.HANDOFFS})._k2_keys(rows, 5)[7]
    assert off == k2.K2Build(tokens=rows, nblocks=3, qlen=5)


def test_a_default_build_keeps_its_symbol():
    key = k2.K2Build(tokens=25, nblocks=3)
    assert k2.symbol(key) == "_ZN4atom42k3_mono_k2_s25_nb3_r1_q1_sp0_dk0_up0_g0_t0E"
    sym = k2.symbol(replace(key, latq_first=True, ln_sq=True))
    assert sym == "_ZN4atom50k3_mono_k2_s25_nb3_r1_q1_sp0_dk0_up0_lf1_lq1_g0_t0E"


@pytest.mark.parametrize("extra", [("rlogp",), ("lsq",), ("rlogp", "lsq")])
def test_opt_in_peer_regions_sit_past_the_default_ones(extra):
    for s in range(1, MAX_ROWS + 1):
        base, lay = k2.peer_layout(s), k2.peer_layout(s, extra)
        end = max(o + n for o, n in base.values())
        assert {r: lay[r] for r in base} == base
        assert set(lay) - set(base) == set(extra)
        assert all(lay[r][0] >= end for r in extra)
        assert k2.peer_bytes(s, extra) >= k2.peer_bytes(s)
    off = runner.ix_peer_offset(MAX_ROWS, extra)
    assert off >= max(k2.peer_bytes(s, extra) for s in range(1, MAX_ROWS + 1))
    assert off >= runner.ix_peer_offset(MAX_ROWS)


@pytest.mark.parametrize("s", [1, 5, 20, 64])
def test_route_parts_reads_every_part_the_router_writes(s):
    """RLOGP: the pairs the router's parts write (``early_out``) are the ones
    the route reads (``route_logits``), each once, inside the region."""
    per, rows, parts, tp = moe.E // moe.TP, moe.ROWS, moe.R_PARTS, moe.TP
    wrote = [
        ((rank * parts + q) * s + t) * per + i * rows + r
        for rank in range(tp)
        for q in range(parts)
        for t in range(s)
        for i in range(moe.R_GROUPS)
        for r in range(rows)
    ]
    read = [
        ((e // per * parts + q) * s + t) * per + e % per
        for e in range(moe.E)
        for q in range(parts)
        for t in range(s)
    ]
    assert sorted(wrote) == sorted(read) == list(range(len(read)))
    assert len(read) == moe.peer_pairs(s, ("rlogp",))["rlogp"]


@pytest.mark.parametrize("s", range(1, MAX_ROWS + 1))
def test_the_latent_first_bitmaps_fit_the_stage_union(s):
    """``latq_first``: the bitmaps past the ug member's words (so the latent
    it gathered first survives the route table's build), and the member no
    larger than the early one: the stage union does not grow."""
    ug = dsl_size_of(moe.ug_lds(s))
    assert ug == 4 * moe.ug_words(s)
    mask = dsl_size_of(moe.mask_lds(s))
    assert dsl_size_of(moe.mask_lds(s, True)) == ug + mask
    sg = min(s, moe.TG)
    xcols = max(gemv.slab_width(moe.early_shape(k)[0], sg) for k in (0, 1, 2))
    assert ug + mask <= sg * (xcols + gemv.LDS_PAD) * 2


@pytest.mark.parametrize("s", range(1, MAX_ROWS + 1))
def test_batched_lnorm_waits_on_its_own_half(s):
    """``stage_lnorm_batched`` polls a pass past ``s * 64`` items at the last
    token: that token is in the same half (the last), so a half never waits
    for the other's LMSQ / LSQ."""
    n, halves = s * 64, moe.ln_halves(s)
    npass = moe.passes(n)
    for h in range(halves):
        ps = moe.ln_passes(npass, halves, h)
        tokens = {
            e // 64 for p in ps for e in range(p * THREADS, (p + 1) * THREADS) if e < n
        }
        for p in ps:
            if (p + 1) * THREADS > n:
                assert s - 1 in tokens


@pytest.mark.parametrize("s", range(1, MAX_ROWS + 1))
def test_batched_reduce_covers_reduce_rows(s):
    """``reduce_batched`` over the step: the (token, column) items of
    ``reduce_rows`` group by group, each once."""
    tg = moe.TG
    by_group = sorted(
        (g * tg + e // 64, e % 64)
        for g in range(groups(s))
        for e in range(min(tg, s - g * tg) * 64)
    )
    whole = sorted((e // 64, e % 64) for e in range(s * 64))
    assert by_group == whole
    assert moe.passes(s * 64) <= 8  # one poll batch, 16 VGPRs of pairs


def _resources(launcher):
    (artifact,) = launcher._mem_cache.values()
    return kernel_resources(_code_object(artifact._ir_text))


@pytest.mark.parametrize("rows", [5, 20, 60])
def test_compile_keeps_hand_offs_and_resources(rows, tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    # On a GPU host importing aiter already initializes CUDA (Triton's arch probe).
    cuda_was_initialized = torch.cuda.is_initialized()
    arms = {"off": {}, **{n: {n: True} for n in k2.HANDOFFS}}
    arms["all"] = {n: True for n in k2.HANDOFFS}
    seen = {}
    for arm, flags in arms.items():
        key = k2.K2Build(tokens=rows, nblocks=4, qlen=5, **flags)
        launcher = k2.build(key)
        rec = trace(launcher, k2.ABI)
        assert not rec.unnamed
        access = {(a.stage, a.region, a.space, a.kind) for a in rec.accesses}
        r = _resources(launcher)
        assert r["scratch"] == r["vgpr_spill"] == 0, (arm, r)
        assert r["lds"] <= 163840, (arm, r)
        seen[arm] = (access, r)
    off, r_off = seen["off"]
    for arm in ("latq_first", "batch_polls"):
        assert seen[arm][0] == off
    regions = lambda acc: {a[1] for a in acc}
    parts, _ = seen["route_parts"]
    assert ("k2.early", "rlogp", Space.PEER, "put") in parts
    assert ("k2.route", "rlogp", Space.PEER, "poll") in parts
    assert regions(off) - regions(parts) == {"rpart", "rlog"}
    sq, _ = seen["ln_sq"]
    assert ("k2.lnorm", "lsq", Space.PEER, "put") in sq
    assert ("k2.lnorm", "lsq", Space.PEER, "poll") in sq
    assert regions(off) - regions(sq) == {"lmsq"}
    assert regions(seen["all"][0]) == regions(off) - {"rpart", "rlog", "lmsq"} | {
        "rlogp",
        "lsq",
    }
    for arm, (_, r) in seen.items():
        assert r["lds"] == r_off["lds"], arm
    assert cuda_was_initialized or not torch.cuda.is_initialized()
    print({"rows": rows, **{a: (r["vgpr"], r["lds"]) for a, (_, r) in seen.items()}})
