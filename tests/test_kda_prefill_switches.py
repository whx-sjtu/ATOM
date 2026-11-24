# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""PKDA's prefill switches: their defaults, and 0 / 1 from the environment."""

import pytest

from atom.utils import envs

BIT_EXACT = (
    "ATOM_CONV1D_TILED_PREFILL",
    "ATOM_KDA_PREFILL_PAGED_STATE",
    "ATOM_KDA_PREFILL_K2AB_WIDE",
    "ATOM_KDA_PREFILL_CKPT_SPLIT",
)
CHANGES_NUMERICS = (
    "ATOM_KDA_PREFILL_SEG_BALANCE",
    "ATOM_KDA_PREFILL_SCAN_PREFETCH",
    "ATOM_KDA_PREFILL_CKPT_SPLIT_ALL",
)
OFF_BY_DEFAULT = ()
ON_BY_DEFAULT = tuple(
    name for name in BIT_EXACT + CHANGES_NUMERICS if name not in OFF_BY_DEFAULT
)


@pytest.mark.parametrize("name", BIT_EXACT + CHANGES_NUMERICS)
def test_default_and_override(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is (name in ON_BY_DEFAULT)
    monkeypatch.setenv(name, "0")
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True
