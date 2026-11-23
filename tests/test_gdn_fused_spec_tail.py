# SPDX-License-Identifier: MIT
"""ATOM_GDN_FUSED_SPEC_TAIL writes what the separate launches write.

Two builders run the same spec decode steps, one per path, and every buffer
the step leaves behind -- masks, query start loc, accepted counts, the
ReplaySSM record cursors -- must match bit for bit. Needs a GPU, or a CPU run
under ``TRITON_INTERPRET=1`` (set before import).
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    pytest.skip("needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)


def _import_gdn_attn():
    arch_info = pytest.importorskip("aiter.ops.triton.utils._triton.arch_info")
    lent = arch_info._CACHED_ARCH is None
    if lent:
        arch_info._CACHED_ARCH = "gfx950"
    try:
        return pytest.importorskip("atom.model_ops.attentions.gdn_attn")
    finally:
        if lent:
            arch_info._CACHED_ARCH = None


gdn_attn = _import_gdn_attn()

MAX_BS, NUM_SPEC, SLOTS, CACHE_LEN = 8, 4, 24, 16


def _builder(fused, replay):
    b = object.__new__(gdn_attn.GDNAttentionMetadataBuilder)
    b.device = DEVICE
    b.max_bs = MAX_BS
    b.num_spec = NUM_SPEC
    b.use_spec_decode = True
    b.replayssm = replay
    b.replayssm_max_query_len = NUM_SPEC + 1
    b.replayssm_cache_len = CACHE_LEN
    b.replayssm_route = "auto"
    b._fused_spec_tail = fused
    b._has_state_fork = False
    b._fork_slots = []
    width = NUM_SPEC + 1
    b.spec_sequence_masks = torch.ones(MAX_BS, dtype=torch.bool, device=DEVICE)
    b.spec_token_indx = torch.arange(MAX_BS * width, dtype=torch.int32, device=DEVICE)
    b.non_spec_token_indx = torch.empty(
        MAX_BS * width, dtype=torch.int32, device=DEVICE
    )
    b.spec_query_start_loc = torch.arange(
        0, (MAX_BS + 1) * width, width, dtype=torch.int32, device=DEVICE
    )
    b.non_spec_query_start_loc = torch.arange(
        MAX_BS + 1, dtype=torch.int32, device=DEVICE
    )
    b.num_accepted_tokens = torch.ones(MAX_BS, dtype=torch.int32, device=DEVICE)
    b.model_runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(num_bonus=None),
        replayssm_write_pos=torch.zeros(SLOTS, dtype=torch.int32, device=DEVICE),
        forward_vars={
            "spec_state_indices": SimpleNamespace(
                gpu=torch.zeros(MAX_BS, width, dtype=torch.int32, device=DEVICE)
            ),
            "non_spec_state_indices": SimpleNamespace(
                gpu=torch.zeros(MAX_BS, dtype=torch.int32, device=DEVICE)
            ),
            "non_spec_state_indices_in": SimpleNamespace(
                gpu=torch.zeros(MAX_BS, dtype=torch.int32, device=DEVICE)
            ),
        },
    )
    return b


def _steps():
    """Spec decode steps: a larger batch, a smaller one (stale tails), a full
    one, an empty one; cursors at the prefill sentinel, inside the record
    buffer and at the flush edge; a PAD slot among the live rows."""
    rng = np.random.default_rng(0)
    for n in (5, 2, MAX_BS, 0, 3, MAX_BS, 1):
        width = NUM_SPEC + 1
        slots = rng.permutation(SLOTS)[:MAX_BS].astype(np.int32)
        slots[n:] = -1
        if n > 2:
            slots[1] = -1
        num_bonus = rng.integers(0, NUM_SPEC + 1, size=n).astype(np.int32)
        cursors = rng.integers(-1, CACHE_LEN - width + 3, size=SLOTS).astype(np.int32)
        cursors[slots[: min(n, 1)]] = -1
        cu = np.zeros(MAX_BS + 1, dtype=np.int32)
        cu[1 : n + 1] = np.cumsum(np.full(n, width))
        cu[n + 1 :] = 12345  # never read past the live prefix
        yield n, slots, num_bonus if n else None, cursors, cu


def _run(builder, n, slots, num_bonus, cursors, cu):
    runner = builder.model_runner
    runner.tokenID_processor.num_bonus = num_bonus
    runner.replayssm_write_pos.copy_(torch.from_numpy(cursors))
    builder.non_spec_state_indices_tensor.gpu.copy_(torch.from_numpy(slots))
    builder._fork_slots = []
    width = NUM_SPEC + 1
    batch = SimpleNamespace(
        total_seqs_num=n,
        total_seqs_num_decode=n,
        total_seqs_num_prefill=0,
        total_tokens_num=n * width,
        total_tokens_num_decode=n * width,
        total_tokens_num_prefill=0,
    )
    attn_metadata = SimpleNamespace(
        cu_seqlens_q=torch.from_numpy(cu).to(DEVICE), num_cached_tokens=None
    )
    builder._attach_gdn_decode_metadata(
        batch, attn_metadata, prepare_block_tables=False
    )
    md = attn_metadata.gdn_metadata
    return {
        "masks": builder.spec_sequence_masks.clone(),
        "query_start_loc": builder.spec_query_start_loc.clone(),
        "num_accepted": builder.num_accepted_tokens.clone(),
        "write_pos": runner.replayssm_write_pos.clone(),
        "md.masks": md.spec_sequence_masks.clone(),
        "md.query_start_loc": md.spec_query_start_loc.clone(),
        "md.num_accepted": md.num_accepted_tokens.clone(),
    }


@pytest.mark.parametrize("replay", [True, False])
def test_fused_tail_matches_the_separate_launches(replay):
    fused, separate = _builder(True, replay), _builder(False, replay)
    addresses = [
        t.data_ptr()
        for t in (
            fused.spec_sequence_masks,
            fused.spec_query_start_loc,
            fused.num_accepted_tokens,
        )
    ]
    for step in _steps():
        got, want = _run(fused, *step), _run(separate, *step)
        for name in want:
            assert got[name].dtype == want[name].dtype, name
            assert torch.equal(got[name], want[name]), (name, step[0])
    # Captured graphs read these buffers by address.
    assert addresses == [
        t.data_ptr()
        for t in (
            fused.spec_sequence_masks,
            fused.spec_query_start_loc,
            fused.num_accepted_tokens,
        )
    ]


@pytest.mark.parametrize("fork, mixed", [(True, False), (False, True)])
def test_a_fork_or_a_mixed_batch_keeps_the_separate_launches(monkeypatch, fork, mixed):
    builder = _builder(True, True)
    builder._fork_slots = [(5, 3)] if fork else []
    seen = {}

    def prepare(batch, attn_metadata, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(spec_query_start_loc=attn_metadata.cu_seqlens_q)

    monkeypatch.setattr(builder, "prepare_gdn_metadata", prepare)
    monkeypatch.setattr(
        builder, "_spec_decode_tail", lambda *a: pytest.fail("fused"), raising=False
    )
    batch = SimpleNamespace(total_seqs_num=3 if mixed else 2, total_seqs_num_decode=2)
    attn_metadata = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 5, 10, 11], dtype=torch.int32, device=DEVICE)
    )
    builder._attach_gdn_decode_metadata(
        batch, attn_metadata, prepare_block_tables=False
    )
    assert seen["fused_spec_tail"] is False
