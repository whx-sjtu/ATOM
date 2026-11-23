# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""PKDA's prefill switches: their defaults, and 0 / 1 from the environment."""

import pytest

from atom.utils import envs

BIT_EXACT = (
    "ATOM_CONV1D_TILED_PREFILL",
    "ATOM_KDA_PREFILL_PAGED_STATE",
    "ATOM_KDA_PREFILL_K2AB_WIDE",
)
CHANGES_NUMERICS = ("ATOM_KDA_PREFILL_SEG_BALANCE", "ATOM_KDA_PREFILL_SCAN_PREFETCH")
ON_BY_DEFAULT = BIT_EXACT


@pytest.mark.parametrize("name", BIT_EXACT + CHANGES_NUMERICS)
def test_default_and_override(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is (name in ON_BY_DEFAULT)
    monkeypatch.setenv(name, "0")
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True
