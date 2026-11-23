# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MOE_ELIDE_FULL_ROW_COUNT: aiter's fused_moe gets no device row count
only in an eager prefill whose host count already covers every row."""

import logging
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def moe(atom_modules, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        atom_modules.moe, "ELIDE_ROW_COUNT_STATS", {"elided": 0, "kept": 0}
    )
    return atom_modules.moe


def _step(monkeypatch, moe, context):
    monkeypatch.setattr(
        moe, "get_forward_context", lambda: SimpleNamespace(context=context)
    )


COUNT = torch.tensor([8], dtype=torch.int32)


@pytest.mark.parametrize("flag", ["0", None])
def test_off_by_default(moe, monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("ATOM_MOE_ELIDE_FULL_ROW_COUNT", raising=False)
    else:
        monkeypatch.setenv("ATOM_MOE_ELIDE_FULL_ROW_COUNT", flag)
    _step(monkeypatch, moe, SimpleNamespace(is_prefill=True, scheduled_tokens=8))
    assert moe._fused_moe_row_count(COUNT, 8) is COUNT
    assert moe.ELIDE_ROW_COUNT_STATS == {"elided": 0, "kept": 0}


def test_eager_prefill_with_every_row_real_drops_the_count(moe, monkeypatch):
    monkeypatch.setenv("ATOM_MOE_ELIDE_FULL_ROW_COUNT", "1")
    _step(monkeypatch, moe, SimpleNamespace(is_prefill=True, scheduled_tokens=8))
    assert moe._fused_moe_row_count(COUNT, 8) is None
    assert moe._fused_moe_row_count(None, 8) is None
    assert moe.ELIDE_ROW_COUNT_STATS == {"elided": 1, "kept": 0}


@pytest.mark.parametrize(
    "context, rows, kept",
    [
        (SimpleNamespace(is_prefill=True, scheduled_tokens=6), 8, 1),  # padded
        (SimpleNamespace(is_prefill=False, scheduled_tokens=8), 8, 0),  # decode
        (None, 8, 0),  # outside an ATOM forward
    ],
)
def test_count_kept_unless_every_row_is_real(moe, monkeypatch, context, rows, kept):
    monkeypatch.setenv("ATOM_MOE_ELIDE_FULL_ROW_COUNT", "1")
    _step(monkeypatch, moe, context)
    assert moe._fused_moe_row_count(COUNT, rows) is COUNT
    assert moe.ELIDE_ROW_COUNT_STATS == {"elided": 0, "kept": kept}


def test_count_kept_while_recording(moe, monkeypatch):
    monkeypatch.setenv("ATOM_MOE_ELIDE_FULL_ROW_COUNT", "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _step(monkeypatch, moe, SimpleNamespace(is_prefill=True, scheduled_tokens=8))
    assert moe._fused_moe_row_count(COUNT, 8) is COUNT
    assert moe.ELIDE_ROW_COUNT_STATS == {"elided": 0, "kept": 0}


def test_first_elided_and_first_kept_pass_logged_once(moe, monkeypatch, caplog):
    monkeypatch.setenv("ATOM_MOE_ELIDE_FULL_ROW_COUNT", "1")
    real = SimpleNamespace(is_prefill=True, scheduled_tokens=8)
    padded = SimpleNamespace(is_prefill=True, scheduled_tokens=6)
    with caplog.at_level(logging.INFO, logger="atom"):
        for context in (real, padded, real, padded, real):
            _step(monkeypatch, moe, context)
            moe._fused_moe_row_count(COUNT, 8)
    lines = [
        r.getMessage()
        for r in caplog.records
        if "ELIDE_FULL_ROW_COUNT" in r.getMessage()
    ]
    assert len(lines) == 2
    assert "8-row eager prefill without its row count" in lines[0]
    assert "8-row prefill pass with 6 scheduled tokens" in lines[1]
    assert moe.ELIDE_ROW_COUNT_STATS == {"elided": 3, "kept": 2}
