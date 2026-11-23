# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K1's KDA tail switches (``kda.KDA_TAIL``): a chunk task's q / k from its own
conv of the head (``own_qk``), a task's f_b rows and ReplaySSM records issued
with its other inputs (``early_in``), a head's parts writing their conv state
after the norm's poll (``cs_late``). Each keeps every sum and its order; only
the round trips between them move."""

import re
from dataclasses import fields
from types import SimpleNamespace

import pytest

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from atom.models.kimi_k3.mono import runner
from atom.models.kimi_k3.mono.kernels.attention import kda as k1
from atom.utils import envs

SWITCHES = dict(runner.K1_KDA_TAIL_ENVS)
SYMS = {"own_qk": "oq", "early_in": "ei", "cs_late": "cl"}


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_default_is_on(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name)
    monkeypatch.setenv(name, "0")
    assert not getattr(envs, name)


def test_the_switches_are_build_fields():
    assert tuple(SWITCHES.values()) == k1.KDA_TAIL
    by_name = {f.name: f for f in fields(k1.KdaPreBuild)}
    for name in k1.KDA_TAIL:
        assert by_name[name].default is False
        assert by_name[name].metadata["sym"] == SYMS[name]


@pytest.mark.parametrize("rows", [5, 10, 15, 25, 35, 55, 60])
def test_a_switch_is_set_only_where_it_moves_something(rows):
    head = k1.head_tasks(rows, 5)
    got = k1.kda_tail_at(rows, 5, 16, dict.fromkeys(k1.KDA_TAIL, True))
    assert got == {
        "own_qk": not head,
        "early_in": True,
        "cs_late": head and k1.head_parts(rows, 5) > 1,
    }
    assert k1.kda_tail_at(rows, 5, 16, {}) == dict.fromkeys(k1.KDA_TAIL, False)


def test_a_plain_decode_without_records_has_nothing_to_issue_early():
    assert not k1.kda_tail_at(8, 1, 0, {"early_in": True})["early_in"]
    assert k1.kda_tail_at(8, 1, 16, {"early_in": True})["early_in"]


def _runner(tail):
    obj = object.__new__(runner.K3MonoDecodeRunner)
    obj.k1_tail = tail
    obj.state_len = 7
    obj.replay_cap = 0
    seam = SimpleNamespace(eps=1e-5, out_eps=1e-5)
    kda = SimpleNamespace(
        blocks=3, write_idx=-1, seam=seam, onorm_eps=1e-5, lower_bound=-5.0
    )
    obj.plans = [SimpleNamespace(layer_id=7, mono=True, kda=kda)]
    return obj


@pytest.mark.parametrize("rows", [5, 35])
@pytest.mark.parametrize("on", [(), *[(n,) for n in k1.KDA_TAIL], k1.KDA_TAIL])
def test_keys_carry_the_switches(rows, on):
    flags = {name: name in on for name in k1.KDA_TAIL}
    key = _runner(flags)._k1_keys(rows, 5)[7]
    want = k1.kda_tail_at(rows, 5, 0, flags)
    assert {f: getattr(key, f) for f in k1.KDA_TAIL} == want
    base = _runner({})._k1_keys(rows, 5)[7]
    assert k1.scratch_bytes(key) == k1.scratch_bytes(base)
    assert k1.kda_lds_words(key) == k1.kda_lds_words(base)
    sym = k1.symbol(key)
    for name in k1.KDA_TAIL:
        assert bool(re.search(f"_{SYMS[name]}1(_|E)", sym)) == want[name]
        assert not re.search(f"_{SYMS[name]}0(_|E)", sym)
    if not any(want.values()):
        assert sym == k1.symbol(base)
