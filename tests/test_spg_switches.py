# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""SPG's small-prefill glue switches: their defaults, and 0 / 1 from the
environment."""

import pytest

from atom.utils import envs

BIT_EXACT = (
    "ATOM_SPG_INDEXER_STAGE",
    "ATOM_SPG_KDA_STATE",
    "ATOM_SPG_KDA_PACK",
    "ATOM_SPG_DCP_LSE",
    "ATOM_SPG_REGION_OUT",
)
ON_BY_DEFAULT = BIT_EXACT


@pytest.mark.parametrize("name", BIT_EXACT)
def test_default_and_override(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is (name in ON_BY_DEFAULT)
    monkeypatch.setenv(name, "0")
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True
