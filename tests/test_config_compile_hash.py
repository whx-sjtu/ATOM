# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The compile cache key must change with every value baked into the graph."""

from types import SimpleNamespace

from atom.config import Config


def _hash(**overrides):
    fields = {
        "quant_config": None,
        "compilation_config": None,
        "parallel_config": None,
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "sequence_parallel_size": 1,
        "dcp_config": None,
        "enable_dp_attention": False,
        "index_cache_dtype": None,
        "index_cache_layout": None,
        "hf_config": SimpleNamespace(),
        "max_model_len": 4096,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 16384,
        "attn_prefill_chunk_size": 16384,
        "kv_cache_block_size": 16,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.9,
        "speculative_config": None,
    }
    fields.update(overrides)
    return Config.compute_hash(SimpleNamespace(**fields))


def test_hash_is_stable():
    assert _hash() == _hash()


def test_hash_changes_with_max_model_len():
    # The sparse-attention indexer bakes max_model_len into the graph.
    assert _hash(max_model_len=12288) != _hash()


def test_hash_changes_with_max_num_seqs():
    assert _hash(max_num_seqs=512) != _hash()


def test_hash_changes_with_max_num_batched_tokens():
    # Buffers sized by it reach the graph as fixed-shape inputs; reusing a
    # 16384 artifact at 8192 fails assert_size_stride on the first forward.
    assert _hash(max_num_batched_tokens=8192) != _hash()
    assert _hash(max_num_batched_tokens=4096) != _hash(max_num_batched_tokens=8192)


def test_hash_changes_with_kv_cache_dtype():
    assert _hash(kv_cache_dtype="bf16") != _hash()


def test_hash_keys_the_row_parallel_front(monkeypatch):
    # The front moves the q/kv projections into an opaque op; a front run that
    # loaded a FRONT=0 artifact failed copy_misaligned_inputs. Row-parallel
    # alone stays inside opaque ops and keeps the artifact; without DCP no
    # layer opts in. The DCP size is the top-level field: the native engine's
    # ParallelConfig keeps its default of 1.
    parallel = SimpleNamespace(
        decode_context_parallel_size=1, compute_hash=lambda: "native"
    )

    def hashed(row_parallel, front, dcp=8):
        for name, value in (
            ("ATOM_DSA_ROW_PARALLEL_PREFILL", row_parallel),
            ("ATOM_DSA_ROW_PARALLEL_PREFILL_FRONT", front),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        return _hash(parallel_config=parallel, decode_context_parallel_size=dcp)

    off = hashed("0", "1")
    assert hashed("1", "0") == off
    assert hashed("1", "1") != off
    assert hashed(None, None) == hashed("1", "1")
    assert hashed(None, None, dcp=1) == hashed("0", "1", dcp=1)


def test_hash_keys_the_qkvg_front_only_where_the_front_is(monkeypatch):
    # ATOM_FUSB_QKVG swaps the front op for one that returns g_proj too; the
    # wq_b skip stays inside the op.
    parallel = SimpleNamespace(
        decode_context_parallel_size=1, compute_hash=lambda: "native"
    )
    monkeypatch.delenv("ATOM_DSA_ROW_PARALLEL_PREFILL", raising=False)

    def hashed(qkvg, front="1", dcp=8, skip=None):
        monkeypatch.setenv("ATOM_DSA_ROW_PARALLEL_PREFILL_FRONT", front)
        for name, value in (("ATOM_FUSB_QKVG", qkvg), ("ATOM_FUSB_SKIP_WQB", skip)):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        return _hash(parallel_config=parallel, decode_context_parallel_size=dcp)

    assert hashed("1") != hashed("0")
    assert hashed(None) == hashed("0") == hashed("0", skip="1")
    assert hashed("1", front="0") == hashed("0", front="0")
    assert hashed("1", dcp=1) == hashed("0", dcp=1)


def test_the_native_engine_kwargs_key_the_row_parallel_front(monkeypatch):
    # The kwargs LLMEngine builds Config from (-tp 8 -dcp <dcp>): the DCP
    # size is the top-level one, its ParallelConfig keeps the default.
    from atom.model_engine.arg_utils import EngineArgs

    monkeypatch.delenv("ATOM_DSA_ROW_PARALLEL_PREFILL", raising=False)

    def hashed(dcp, front):
        monkeypatch.setenv("ATOM_DSA_ROW_PARALLEL_PREFILL_FRONT", front)
        kwargs = EngineArgs(
            model="m", tensor_parallel_size=8, decode_context_parallel_size=dcp
        )._get_engine_kwargs()
        return _hash(
            parallel_config=kwargs["parallel_config"],
            decode_context_parallel_size=kwargs["decode_context_parallel_size"],
        )

    assert hashed(8, "1") != hashed(8, "0")
    assert hashed(1, "1") == hashed(1, "0")


def test_hash_ignores_settings_that_stay_out_of_the_graph():
    # These size attention-metadata workspaces, KV pages or the KV pool, all
    # outside the traced graph; changing them must keep the cached artifact.
    base = _hash()
    assert _hash(attn_prefill_chunk_size=8192) == base
    assert _hash(kv_cache_block_size=256) == base
    assert _hash(gpu_memory_utilization=0.8) == base
