# SPDX-License-Identifier: MIT
"""Row-parallel DSA prefill (``ATOM_DSA_ROW_PARALLEL_PREFILL``,
``atom/model_ops/dsa_row_parallel.py``).

The plan: every candidate a rank's rows select reads, from the gathered KV,
the row of the position the top-k named -- simulated here over every rank's
cache, filled where the DCP writer puts each token, and with the front on its
rows (``_FRONT``) over the new tokens each rank wrote after it. The wiring: with the
switch off, or on a step it does not take, the builder, the indexer and
sparse MLA run exactly the calls they ran before it existed; on a step it
takes, each runs the row-parallel calls over this rank's rows. The GPU
cases check the kernels against the references the CPU cases use.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a GPU"
)

rp = aiter_mla = attention_mla = indexer_forward = backends = None


@pytest.fixture(scope="module", autouse=True)
def _modules():
    global rp, aiter_mla, attention_mla, indexer_forward, backends
    with atom_without_aiter() as load:
        rp = load("atom.model_ops.dsa_row_parallel")
        aiter_mla = load("atom.model_ops.attentions.aiter_mla")
        attention_mla = load("atom.model_ops.attention_mla")
        indexer_forward = load("atom.model_ops.indexer.indexer_forward")
        backends = load("atom.model_ops.attentions.backends")
        yield


class _Stop(Exception):
    pass


# ───────────────────────────────────────── the layout, by reference ──


def owner_slot(block_table, pos, world, block):
    """Where DCP keeps position ``pos``: its owner and the owner's cache slot
    (the prefill writer's and the owned-slot filter's mapping)."""
    local = pos // world
    return pos % world, int(block_table[local // block]) * block + local % block


def ref_index(indptr, batch_ids, topk, shard_base, rows_pad, world, out):
    """The index kernel's mapping, row by row."""
    for t in range(len(batch_ids)):
        req, start = int(batch_ids[t]), int(indptr[t])
        for c in range(int(indptr[t + 1]) - start):
            p = int(topk[t, c])
            base, end = int(shard_base[req]), int(shard_base[req + 1])
            local = p // world
            ok = p >= 0 and base + local < end
            out[start + c] = (p % world) * rows_pad + base + local if ok else 0
    return out


def _requests(context_lens, world, block, seed=0):
    """Block tables over a shuffled pool, each request's local blocks."""
    rng = np.random.default_rng(seed)
    need = [-(-n // (world * block)) for n in context_lens]
    pool = rng.permutation(sum(need) * 2)
    tables, at = [], 0
    for n in need:
        tables.append(pool[at : at + n].astype(np.int32).tolist())
        at += n
    return tables, int(pool.max()) + 1


def _gathered(context_lens, tables, world, block, n_blocks, plan):
    """Each rank's cache filled with its tokens' global ids (request, position),
    then gathered rank-major through the plan: what `gather_kv` returns."""
    caches = [np.full((n_blocks * block, 2), -1, dtype=np.int64) for _ in range(world)]
    for i, n in enumerate(context_lens):
        for p in range(n):
            owner, slot = owner_slot(tables[i], p, world, block)
            caches[owner][slot] = (i, p)
    slots = plan.slot_ids.numpy()
    return np.concatenate([c[slots] for c in caches])


@pytest.mark.parametrize("world,block", [(8, 64), (4, 4), (2, 1)])
@pytest.mark.parametrize("context_lens", [[300], [1, 129, 70], [64 * 8 * 3]])
def test_every_candidate_reads_the_row_of_its_position(world, block, context_lens):
    tables, n_blocks = _requests(context_lens, world, block)
    plan = rp.gathered_kv_plan(np.array(context_lens), tables, world, block, "cpu")
    assert plan.rows_pad * world == rp.gathered_kv_rows(context_lens, world, block)
    gathered = _gathered(context_lens, tables, world, block, n_blocks, plan)
    rng = np.random.default_rng(1)
    topk_width = 16
    batch_ids, rows = [], []
    for i, n in enumerate(context_lens):
        for _ in range(3):
            visible = int(rng.integers(1, n + 1))
            k = min(visible, topk_width)
            row = np.full(topk_width, -1, dtype=np.int32)
            row[:k] = rng.choice(visible, size=k, replace=False)
            batch_ids.append(i)
            rows.append(row)
    topk = np.stack(rows)
    counts = (topk >= 0).sum(1)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    out = ref_index(
        indptr,
        batch_ids,
        topk,
        plan.shard_base.numpy(),
        plan.rows_pad,
        world,
        np.full(indptr[-1], -7, dtype=np.int64),
    )
    for t, (i, row) in enumerate(zip(batch_ids, rows)):
        for c in range(counts[t]):
            assert tuple(gathered[out[indptr[t] + c]]) == (i, int(row[c]))


def test_padded_lengths_are_whole_local_blocks_on_every_rank():
    lens = rp.padded_local_lens([1, 512, 513, 0], 8, 64)
    assert lens.tolist() == [64, 64, 128, 0]
    assert rp.gathered_kv_rows([1, 512, 513], 8, 64) == 8 * 256


def test_a_plan_refuses_a_block_table_too_short():
    with pytest.raises(ValueError):
        rp.gathered_kv_plan(np.array([1025]), [[0]], 8, 64, "cpu")
    with pytest.raises(ValueError):
        rp.gathered_kv_plan(np.array([10, 10]), [[0]], 8, 64, "cpu")


def test_own_rows_take_their_indptr_rebased():
    host = np.array([0, 3, 7, 7, 12, 20], dtype=np.int32)
    own, base = rp.own_kv_indptr(host, 2, 4)
    assert base == 7 and own.tolist() == [0, 0, 5] and own.dtype == np.int32


@pytest.mark.parametrize("tokens", [8192, 8193, 16384, 16383])
def test_the_rows_split_as_the_indexer_splits_them(tokens):
    from atom.model_ops.indexer import RowSplit

    world = 8
    covered = []
    for r in range(world):
        split = RowSplit.of(tokens, r, world)
        assert split.slot == split.first == min(r * split.stride, tokens)
        covered.extend(range(split.first, split.end))
    assert covered == list(range(tokens))


def test_rows_go_back_to_heads_in_row_order(monkeypatch):
    """Every rank's rows of all heads, all-to-all'd: each rank gets every row
    of its heads, rows past the step undefined and cut or padded away."""
    world, h, v, tokens = 4, 2, 3, 10
    stride = -(-tokens // world)
    full = torch.arange(tokens * world * h * v, dtype=torch.float32).view(
        tokens, world * h, v
    )
    sends = []
    for r in range(world):
        first, end = min(r * stride, tokens), min((r + 1) * stride, tokens)
        p = torch.full((world * h, stride, v), -1.0)
        p[:, : end - first] = full[first:end].transpose(0, 1)
        sends.append(p.view(world, h, stride, v))
    for r in range(world):

        def a2a(recv, send, group=None, r=r):
            for s in range(world):
                recv[s] = sends[s][r]

        monkeypatch.setattr(torch.distributed, "all_to_all_single", a2a)
        group = SimpleNamespace(world_size=world, device_group=None)
        mine = full[:, r * h : (r + 1) * h].reshape(tokens, h * v)
        for rows in (tokens, tokens + 5, tokens - 1):
            got = rp.rows_to_heads(sends[r].view(world * h, stride, v), rows, group)
            assert got.shape == (rows, h * v)
            n = min(rows, tokens)
            assert torch.equal(got[:n], mine[:n])


def test_the_scratch_is_the_gather_at_the_cap():
    assert rp.scratch_bytes(1 << 18, 8, 576) == ((1 << 18) + (1 << 15)) * 576
    assert rp.scratch_bytes(9, 8, 1) == 9 + 2


_OK = {
    "is_sparse": True,
    "dcp_world_size": 8,
    "kv_cache_dtype": "fp8",
    "mla_page_size": 1,
    "indexer_fp4": True,
    "index_kpool": 1,
    "persistent_plan": True,
}


@pytest.mark.parametrize(
    "change",
    [
        {"is_sparse": False},
        {"dcp_world_size": 1},
        {"kv_cache_dtype": "auto"},
        {"mla_page_size": 64},
        {"indexer_fp4": False},
        {"index_kpool": 4},
        {"persistent_plan": False},
    ],
)
def test_a_builder_that_cannot_run_it_says_why(monkeypatch, change):
    monkeypatch.setattr("atom.plugin.is_plugin_mode", lambda: False)
    monkeypatch.setattr("atom.distributed.pcp_utils.pcp_is_enabled", lambda: False)
    assert rp.builder_blocker(**_OK) is None
    assert rp.builder_blocker(**{**_OK, **change})


@pytest.mark.parametrize("plugin,pcp", [(True, False), (False, True)])
def test_plugins_and_pcp_keep_the_default_path(monkeypatch, plugin, pcp):
    monkeypatch.setattr("atom.plugin.is_plugin_mode", lambda: plugin)
    monkeypatch.setattr("atom.distributed.pcp_utils.pcp_is_enabled", lambda: pcp)
    assert rp.builder_blocker(**_OK)


@pytest.mark.parametrize(
    "name,value", [("MIN_TOKENS", "0"), ("MAX_KV_ROWS", "0"), ("MAX_KV_ROWS", "-1")]
)
def test_a_bad_bound_is_refused(monkeypatch, name, value):
    monkeypatch.setenv(f"ATOM_DSA_ROW_PARALLEL_PREFILL_{name}", value)
    read = rp.min_tokens_env if name == "MIN_TOKENS" else rp.max_kv_rows_env
    with pytest.raises(ValueError):
        read()


def test_the_switch_is_on_by_default(monkeypatch):
    from atom.utils import envs
    from atom.utils.forward_context import AttentionMetaData

    monkeypatch.delenv("ATOM_DSA_ROW_PARALLEL_PREFILL", raising=False)
    assert envs.ATOM_DSA_ROW_PARALLEL_PREFILL is True
    assert rp.requested() is False
    assert envs.ATOM_DSA_ROW_PARALLEL_PREFILL_MIN_TOKENS == 8192
    assert envs.ATOM_DSA_ROW_PARALLEL_PREFILL_MAX_KV_ROWS == 1 << 18
    assert AttentionMetaData().dsa_row_parallel is None
    assert aiter_mla.AiterMLAMetadataBuilder._dsa_row_parallel is False


# ───────────────────────────────────────────── the builder ──

WORLD, BLOCK, TOPK = 8, 64, 2048


def _batch(context_lens, seqlens_q, dummy=False, tables=None):
    from atom.model_ops.attentions.token_layout.prefill import PrefillBatch

    if tables is None:
        tables, _ = _requests(context_lens, WORLD, BLOCK)
    return PrefillBatch(
        context_lens=np.asarray(context_lens, dtype=np.int32),
        seqlens_q=np.asarray(seqlens_q, dtype=np.int32),
        block_tables=tables,
        is_dummy_run=dummy,
    )


def _host_indptr(batch):
    counts = []
    for n, q in zip(batch.context_lens.tolist(), batch.seqlens_q.tolist()):
        counts.extend(min(TOPK, n - q + t + 1) for t in range(q))
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)


def _builder(monkeypatch, *, on=True, rank=3, min_tokens=8192, cap=1 << 18):
    b = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    host = np.zeros(1 << 16, dtype=np.int32)
    work = {
        name: torch.zeros(64 if name == "work_indptr" else 8, dtype=torch.int32)
        for name in backends.PERSISTENT_WORK_BUFFERS
    }
    b.model_runner = SimpleNamespace(
        forward_vars={
            "sparse_kv_indptr": SimpleNamespace(np=host),
            **{f"sparse_prefill_{n}": t for n, t in work.items()},
        },
        block_size=BLOCK,
    )
    b.device = "cpu"
    b.dcp_rank, b.dcp_world_size = rank, WORLD
    b.dcp_sparse_prefill_nonps = False
    b.sparse_prefill_num_heads = 96
    b._sparse_prefill_kv_granularity = TOPK
    b._sparse_prefill_staged = {}
    b._dsa_row_parallel = on
    b._dsa_row_parallel_min_tokens = min_tokens
    b._dsa_row_parallel_max_kv_rows = cap
    b._dsa_row_parallel_layers = True
    b._whole_row_plan_checked = True
    calls = []
    for name in (
        "_dcp_pass_gathers",
        "_build_dcp_indexer_prefill_meta",
        "_sparse_prefill_work_plan",
        "_finish_prefill_index_rows",
        "_apply_pcp_reindex",
    ):
        monkeypatch.setattr(
            b, name, lambda *a, _n=name, **k: calls.append((_n, a, k)), raising=False
        )
    monkeypatch.setattr(b, "_prefill_reads_token_csr", lambda md: False, raising=False)
    monkeypatch.setattr(b, "_has_dense_prefill_layer", lambda: False, raising=False)
    monkeypatch.setattr(
        backends.CommonAttentionBuilder, "_finish_prefill_batch", lambda *a, **k: None
    )
    plans = []
    monkeypatch.setattr(
        aiter_mla, "plan_whole_rows", lambda *a: plans.append(("rows", a))
    )
    monkeypatch.setattr(
        aiter_mla,
        "_aiter_sparse_prefill_plan",
        lambda *a: plans.append(("aiter", a)),
    )
    return b, calls, plans


def _staged(batch, passes=None, pcp=False):
    tokens = int(batch.seqlens_q.sum())
    counts = np.diff(_host_indptr(batch))
    return aiter_mla.StagedSparsePrefill(
        sparse_counts=counts,
        total_kv=int(batch.context_lens.sum()),
        pcp_split=pcp,
        index_rows=aiter_mla.StagedIndexRows(rows=tokens, width=1, passes=passes),
    )


def _metadata(tokens, ubatches=None):
    from atom.utils.forward_context import AttentionMetaData

    md = AttentionMetaData(
        has_cached=True,
        sparse_cu_seqlens_q=torch.arange(tokens + 1, dtype=torch.int32),
        sparse_kv_last_page_lens=torch.ones(tokens, dtype=torch.int32),
    )
    md.run_sparse, md.ubatches = True, ubatches
    return md


def _finish(b, batch, staged, md):
    b.model_runner.forward_vars["sparse_kv_indptr"].np[
        : len(staged.sparse_counts) + 1
    ] = _host_indptr(batch)
    b._sparse_prefill_staged[""] = staged
    b._finish_prefill_batch(batch, md, "")


def _names(calls):
    return [name for name, _, _ in calls]


_DEFAULT = [
    "_build_dcp_indexer_prefill_meta",
    "_sparse_prefill_work_plan",
    "_finish_prefill_index_rows",
]


@pytest.mark.parametrize(
    "on,batch,staged_kw,md_kw,cap",
    [
        (False, ([16384], [16384]), {}, {}, 1 << 18),  # switch off
        (True, ([8191], [8191]), {}, {}, 1 << 18),  # under the bound
        (True, ([16384], [16384], True), {}, {}, 1 << 18),  # a dummy run
        (True, ([16384], [16384]), {}, {"ubatches": [1]}, 1 << 18),  # TBO
        (True, ([16384], [16384]), {"pcp": True}, {}, 1 << 18),
        (True, ([16384], [16384]), {"passes": (1,)}, {}, 1 << 18),  # FP8 passes
        (True, ([16384 + 131072], [16384]), {}, {}, 1 << 17),  # over the cap
    ],
)
def test_a_step_it_does_not_take_runs_the_default_calls(
    monkeypatch, on, batch, staged_kw, md_kw, cap
):
    b, calls, plans = _builder(monkeypatch, on=on, cap=cap)
    batch = _batch(*batch)
    md = _metadata(int(batch.seqlens_q.sum()), **md_kw)
    _finish(b, batch, _staged(batch, **staged_kw), md)
    expected = list(_DEFAULT)
    if staged_kw.get("passes"):
        expected[0] = "_dcp_pass_gathers"
    if staged_kw.get("pcp"):
        expected.insert(2, "_apply_pcp_reindex")
    assert _names(calls) == expected
    _name, args, _ = calls[1]
    assert args == (md, "", 96, md.sparse_kv_last_page_lens)
    assert md.dsa_row_parallel is None and plans == []


@pytest.mark.parametrize(
    "sizes,taken",
    [
        ((), True),
        ((256, 512, *range(1024, 4097, 256)), True),  # Kimi-K3's own
        ((256, 4096, 16384), False),
        ((8192, 32768), False),
    ],
)
def test_a_step_a_token_graph_bucket_holds_keeps_the_default_path(
    monkeypatch, sizes, taken
):
    monkeypatch.setitem(aiter_mla.TokenGraph._sizes, "prefill", sizes)
    b, calls, plans = _builder(monkeypatch)
    batch = _batch([16384], [16384])
    md = _metadata(16384)
    _finish(b, batch, _staged(batch), md)
    assert (md.dsa_row_parallel is not None) is taken
    if not taken:
        assert _names(calls) == _DEFAULT and plans == []


def test_layers_not_ready_keep_the_default_path(monkeypatch):
    b, calls, _plans = _builder(monkeypatch)
    b._dsa_row_parallel_layers = False
    batch = _batch([16384], [16384])
    md = _metadata(16384)
    _finish(b, batch, _staged(batch), md)
    assert _names(calls) == _DEFAULT and md.dsa_row_parallel is None


@pytest.mark.parametrize("rank", [0, 3, 7])
@pytest.mark.parametrize("ctx,tokens", [(0, 16384), (114688, 16384), (0, 8195)])
def test_a_step_it_takes_plans_this_ranks_rows(monkeypatch, rank, ctx, tokens):
    b, calls, plans = _builder(monkeypatch, rank=rank)
    batch = _batch([ctx + tokens], [tokens])
    md = _metadata(tokens)
    _finish(b, batch, _staged(batch), md)
    assert _names(calls) == [
        "_build_dcp_indexer_prefill_meta",
        "_finish_prefill_index_rows",
    ]
    plan = md.dsa_row_parallel
    stride = -(-tokens // WORLD)
    first, end = min(rank * stride, tokens), min((rank + 1) * stride, tokens)
    assert (plan.num_tokens, plan.first, plan.end, plan.stride) == (
        tokens,
        first,
        end,
        stride,
    )
    host = _host_indptr(batch)
    assert plan.kv_base == host[first]
    assert plan.kv_indptr.tolist() == (host[first : end + 1] - host[first]).tolist()
    assert plan.cu_seqlens_q.tolist() == list(range(end - first + 1))
    assert plan.last_page_lens.tolist() == [1] * (end - first)
    assert plan.no_slots.tolist() == [-1] * (end - first)
    assert plan.kv.rows_pad * WORLD == rp.gathered_kv_rows([ctx + tokens], WORLD, BLOCK)
    # One whole-row plan of this rank's rows, into the step's buffers.
    assert [p[0] for p in plans] == ["rows"]
    kv_indptr, rows, _, work = plans[0][1]
    assert kv_indptr is plan.kv_indptr and rows == end - first
    assert work is plan.work
    assert all(
        work[n] is b.model_runner.forward_vars[f"sparse_prefill_{n}"] for n in work
    )


def test_without_the_row_plan_it_plans_with_aiter(monkeypatch):
    monkeypatch.setenv("ATOM_MLA_SPARSE_PREFILL_ROW_PLAN", "0")
    b, _calls, plans = _builder(monkeypatch)
    batch = _batch([16384], [16384])
    md = _metadata(16384)
    _finish(b, batch, _staged(batch), md)
    assert [p[0] for p in plans] == ["aiter"]
    _, meta, heads, last, _work = plans[0][1]
    assert meta.sparse_kv_indptr is md.dsa_row_parallel.kv_indptr
    assert meta.sparse_cu_seqlens_q is md.dsa_row_parallel.cu_seqlens_q
    assert heads == 96 and last is md.dsa_row_parallel.last_page_lens


def test_the_first_whole_row_plan_is_checked_against_aiters(monkeypatch):
    b, _calls, _plans = _builder(monkeypatch)
    b._whole_row_plan_checked = False
    checked = []
    monkeypatch.setattr(
        aiter_mla, "_check_whole_row_plan", lambda *a: checked.append(a)
    )
    batch = _batch([16384], [16384])
    md = _metadata(16384)
    _finish(b, batch, _staged(batch), md)
    ((builder, _meta, _heads, _last, work, ref),) = checked
    assert builder is b and work is md.dsa_row_parallel.work and set(ref) == set(work)


def test_the_switch_off_resolves_nothing(monkeypatch):
    monkeypatch.setenv("ATOM_DSA_ROW_PARALLEL_PREFILL", "0")
    b = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    warmed = []
    monkeypatch.setattr(rp, "warm_gathered_kv_index", lambda *a: warmed.append(a))
    assert b._resolve_dsa_row_parallel(None) is False and warmed == []


def _resolvable(monkeypatch, **kw):
    monkeypatch.setenv("ATOM_DSA_ROW_PARALLEL_PREFILL", "1")
    b = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    b.is_sparse, b.dcp_world_size, b.block_size = True, 8, 1
    b._indexer_fp4, b.index_kpool, b.index_topk_out = True, 1, TOPK
    b.sparse_dcp_metadata_rebuild, b.dcp_sparse_prefill_nonps = True, False
    b.device = "cpu"
    for k, v in kw.items():
        setattr(b, k, v)
    config = SimpleNamespace(
        kv_cache_dtype="fp8", dcp_config=SimpleNamespace(interleave_size=1)
    )
    monkeypatch.setattr("atom.plugin.is_plugin_mode", lambda: False)
    monkeypatch.setattr("atom.distributed.pcp_utils.pcp_is_enabled", lambda: False)
    warmed, warned = [], []
    monkeypatch.setattr(
        rp, "warm_gathered_kv_index", lambda *a, **k: warmed.append((a, k))
    )
    monkeypatch.setattr(aiter_mla.logger, "warning", lambda *a: warned.append(a))
    return b, config, warmed, warned


@pytest.mark.parametrize("front", ["0", "1"])
def test_the_switch_on_warms_the_index_kernel(monkeypatch, front):
    b, config, warmed, warned = _resolvable(monkeypatch)
    monkeypatch.setenv("ATOM_DSA_ROW_PARALLEL_PREFILL_MIN_TOKENS", "4096")
    monkeypatch.setenv("ATOM_DSA_ROW_PARALLEL_PREFILL_FRONT", front)
    assert b._resolve_dsa_row_parallel(config) is True
    assert warmed == [((8, 1, TOPK, "cpu"), {"new_rows": front == "1"})]
    assert warned == [] and b._dsa_row_parallel_front is (front == "1")
    assert b._dsa_row_parallel_min_tokens == 4096
    assert b._dsa_row_parallel_max_kv_rows == 1 << 18


def test_the_switch_on_where_it_cannot_run_warns(monkeypatch):
    b, config, warmed, warned = _resolvable(monkeypatch, dcp_sparse_prefill_nonps=True)
    assert b._resolve_dsa_row_parallel(config) is False
    assert warmed == [] and len(warned) == 1


def test_the_default_where_it_cannot_run_only_logs(monkeypatch):
    b, config, warmed, warned = _resolvable(monkeypatch, dcp_world_size=1)
    monkeypatch.delenv("ATOM_DSA_ROW_PARALLEL_PREFILL")
    logged = []
    monkeypatch.setattr(aiter_mla.logger, "info", lambda *a: logged.append(a))
    assert b._resolve_dsa_row_parallel(config) is False
    assert warmed == [] and warned == []
    assert [a[1:] for a in logged] == [("DCP is off",)]


def _impl(**kw):
    impl = object.__new__(attention_mla.MLAAttention)
    torch.nn.Module.__init__(impl)
    ready = {
        "layer_num": 5,
        "is_sparse_mla": True,
        "dsa_row_parallel": True,
        "dcp_world_size": 8,
        "qrep_enabled": True,
        "_absorb_q_pack_enabled": True,
        "_q_tail_cache": True,
        "pbm_enabled": True,
        "W_V_dcp_scale": None,
        "dcp_sparse_prefill_persistent": True,
        "dcp_sparse_prefill_nonps": False,
        "use_seg_mla": False,
        "kv_shuffled": False,
        "_fused_q_head_pad": False,
        "kv_cache_dtype": "fp8",
    }
    for k, v in {**ready, **kw}.items():
        object.__setattr__(impl, k, v)
    return impl


@pytest.mark.parametrize(
    "change",
    [
        {"dsa_row_parallel": False},
        {"qrep_enabled": False},
        {"_absorb_q_pack_enabled": False},
        {"_q_tail_cache": False},
        {"pbm_enabled": False},
        {"W_V_dcp_scale": torch.ones(1)},
        {"dcp_sparse_prefill_persistent": False},
        {"dcp_sparse_prefill_nonps": True},
        {"use_seg_mla": True},
        {"kv_shuffled": True},
        {"_fused_q_head_pad": True},
        {"kv_cache_dtype": "auto"},
    ],
)
def test_a_layer_that_cannot_run_it_says_why(change):
    assert _impl().dsa_row_parallel_blocker() is None
    assert "layer 5" in _impl(**change).dsa_row_parallel_blocker()


@pytest.mark.parametrize(
    "change,ready", [({}, True), ({"dsa_row_parallel": False}, False)]
)
def test_every_sparse_layer_must_be_ready(monkeypatch, change, ready):
    b = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    dense = _impl(is_sparse_mla=False, dsa_row_parallel=False)
    layers = [_impl(), dense, _impl(**change)]
    b.model_runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: iter(layers))
    )
    b._dsa_row_parallel_layers = None
    monkeypatch.setenv("ATOM_DSA_ROW_PARALLEL_PREFILL", "1")
    warned = []
    monkeypatch.setattr(aiter_mla.logger, "warning", lambda *a: warned.append(a))
    assert b._dsa_row_parallel_layers_ready() is ready
    assert len(warned) == (0 if ready else 1)
    layers.clear()
    assert b._dsa_row_parallel_layers_ready() is ready  # decided once


# ───────────────────────────────────────────── the indexer ──


def _plan(tokens=40, rank=1, world=4):
    split_stride = -(-tokens // world)
    first, end = rank * split_stride, min((rank + 1) * split_stride, tokens)
    return rp.DsaRowParallel(
        num_tokens=tokens,
        first=first,
        end=end,
        stride=split_stride,
        kv=rp.GatheredKV(
            torch.zeros(0, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), 8
        ),
        kv_base=7,
        kv_indptr=torch.zeros(end - first + 1, dtype=torch.int32),
        cu_seqlens_q=torch.arange(end - first + 1, dtype=torch.int32),
        last_page_lens=torch.ones(end - first, dtype=torch.int32),
        no_slots=torch.full((end - first,), -1, dtype=torch.int64),
        work={"work_indptr": torch.zeros(3, dtype=torch.int32)},
    )


def _index_rows(plan, world=4, rank=1):
    from atom.model_ops.indexer import RowSplit

    split = RowSplit.of(plan.num_tokens, rank, world)
    return SimpleNamespace(
        split=split,
        out_rows=split.stride * world,
        passes=None,
        _replace=lambda **kw: SimpleNamespace(split=kw["split"]),
    )


def _indexer_call(monkeypatch, plan, row_parallel=True, tokens=40, topk=128):
    f = indexer_forward
    md = SimpleNamespace(
        slot_mapping=torch.zeros(tokens, dtype=torch.int64),
        run_sparse=True,
        index_rows=(
            _index_rows(plan) if plan is not None else SimpleNamespace(passes=None)
        ),
        dsa_row_parallel=plan,
    )
    ctx = SimpleNamespace(is_dummy_run=False, is_prefill=True, forward_mode=None)
    monkeypatch.setattr(
        f, "get_forward_context", lambda: SimpleNamespace(attn_metadata=md, context=ctx)
    )
    indexer = SimpleNamespace(
        _indexer_fp4=False,
        _index_cache_layout=None,
        use_nope_cache_fusion=False,
        exchange_stream=object(),
        row_parallel=row_parallel,
    )
    cfg = SimpleNamespace(
        kv_cache_block_size=64,
        dcp_config=SimpleNamespace(interleave_size=1),
        compilation_config=SimpleNamespace(static_forward_context={"L": indexer}),
    )
    monkeypatch.setattr(f, "get_current_atom_config", lambda: cfg)
    monkeypatch.setattr(f, "get_dcp_world_size", lambda: 4)
    monkeypatch.setattr(f, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(f, "indexer_k_quant_and_cache", lambda *a, **k: None)
    monkeypatch.setattr(f, "_exchange_stream", lambda s: s)
    seen = []
    monkeypatch.setattr(f, "_row_parallel_select", lambda *a: seen.append(("rp", a)))
    monkeypatch.setattr(
        f, "_dcp_prefill_exchange", lambda *a: seen.append(("exchange", a))
    )
    t = torch.zeros(tokens, 8)
    f.sparse_attn_indexer(
        t,
        "L.k_cache",
        torch.zeros(64, 8),
        t,
        t,
        torch.zeros(tokens, 4),
        128,
        None,
        topk,
        8,
        1 << 16,
        0,
        None,
        None,
        None,
        None,
        None,
        1e-6,
        None,
        None,
        None,
        1.0,
        False,
        False,
        False,
    )
    return seen, md


def test_an_indexer_without_a_plan_runs_the_exchange(monkeypatch):
    seen, _ = _indexer_call(monkeypatch, None)
    assert [s[0] for s in seen] == ["exchange"]


def test_an_indexer_with_a_plan_selects_row_parallel(monkeypatch):
    plan = _plan()
    seen, md = _indexer_call(monkeypatch, plan)
    assert [s[0] for s in seen] == ["rp"]
    args = seen[0][1]
    assert args[5] is md and args[6] is plan and args[8] == 128


def _select(monkeypatch, plan, row_parallel=True, rows=None):
    f = indexer_forward
    tokens, topk = plan.num_tokens, 128
    scored, indexed = [], []

    def score_topk(query, weights, cache, rows, *, topk, weight_scale, stable, out):
        scored.append(rows)
        out[rows.split.first : rows.split.end] = 5
        return out

    monkeypatch.setattr(f, "score_topk", score_topk)
    monkeypatch.setattr(f, "_index_cache", lambda *a: "cache")
    monkeypatch.setattr(f, "get_dcp_world_size", lambda: 4)
    monkeypatch.setattr(f, "gathered_kv_index", lambda *a, **k: indexed.append((a, k)))
    md = SimpleNamespace(
        index_rows=rows or _index_rows(plan),
        sparse_kv_indptr=torch.arange(tokens + 1, dtype=torch.int32),
        batch_id_per_q_token=torch.zeros(tokens, dtype=torch.int32),
        sparse_kv_indices=torch.zeros(tokens * topk, dtype=torch.int32),
    )
    indexer = SimpleNamespace(_indexer_fp4=False, row_parallel=row_parallel)
    topk_indices = torch.full((tokens + 3, topk), -1, dtype=torch.int32)
    f._row_parallel_select(
        indexer, "q", None, "w", "kv", md, plan, topk_indices, topk, 1.0, False, 1
    )
    return scored, indexed, md


def test_row_parallel_selection_scores_and_maps_only_its_rows(monkeypatch):
    plan = _plan()
    scored, indexed, md = _select(monkeypatch, plan)
    (rows,) = scored
    assert rows.split.world == 1
    assert (rows.split.first, rows.split.end) == (plan.first, plan.end)
    (((indptr, batch_ids, topk_rows, kv, world), kw),) = indexed
    assert indptr.tolist() == list(range(plan.first, plan.end + 1))
    assert batch_ids.shape[0] == topk_rows.shape[0] == plan.rows
    assert (topk_rows == 5).all() and kv is plan.kv and world == 4
    assert kw["out"] is md.sparse_kv_indices and kw["num_topk_tokens"] == 128


def test_an_indexer_that_did_not_opt_in_refuses_the_step(monkeypatch):
    with pytest.raises(RuntimeError):
        _select(monkeypatch, _plan(), row_parallel=False)


def test_a_plan_of_other_rows_than_the_indexers_is_refused(monkeypatch):
    plan = _plan(rank=1)
    with pytest.raises(AssertionError):
        _select(monkeypatch, plan, rows=_index_rows(plan, rank=2))


# ───────────────────────────────────────────── sparse MLA ──

H, L, R, V, W = 8, 16, 8, 4, 4  # 8 group heads, 2 a rank


def _mla(monkeypatch, plan, *, dummy=False, opted_in=True, tokens=None):
    tokens = plan.num_tokens if plan is not None else tokens
    impl = _impl(dsa_row_parallel=opted_in, dcp_world_size=W)
    for k, v in {
        "kv_lora_rank": L,
        "qk_rope_head_dim": R,
        "qrep_num_heads": H,
        "num_heads": H // W,
        "dcp_sparse_prefill_num_heads": H,
        "dtype": torch.bfloat16,
        "scale": 0.5,
        "_k_scale": torch.tensor(1.0),
        "_q_scale": torch.tensor(1.0),
        "W_K_qrep": "W_K_qrep",
        "W_V_dcp": torch.zeros(H, V, L),
        "dcp_group": "group",
        "o_proj": torch.nn.Identity(),
    }.items():
        object.__setattr__(impl, k, v)
    md = SimpleNamespace(
        run_sparse=True,
        dsa_row_parallel=plan,
        slot_mapping=torch.arange(tokens, dtype=torch.int64),
        sparse_kv_indices=torch.arange(1000, dtype=torch.int32),
        dtype_q=torch.float8_e4m3fn,
    )
    ctx = SimpleNamespace(is_dummy_run=dummy, is_prefill=True)
    kv_cache = torch.zeros(64, 1, L + R, dtype=torch.float8_e4m3fn)
    monkeypatch.setattr(
        attention_mla,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=md,
            context=ctx,
            kv_cache_data={"layer_5": SimpleNamespace(k_cache=kv_cache)},
        ),
    )
    return impl, md, kv_cache


def _inputs(tokens):
    return (
        torch.zeros(tokens, 12, dtype=torch.bfloat16),
        torch.zeros(tokens, L, dtype=torch.bfloat16),
        torch.zeros(tokens, R, dtype=torch.bfloat16),
    )


def test_a_sparse_prefill_without_a_plan_runs_the_default_path(monkeypatch):
    impl, _md, _ = _mla(monkeypatch, None, tokens=40)

    def q_proj(self, x, x_scale=None, group=False, k_up_proj=True):
        raise _Stop

    monkeypatch.setattr(attention_mla.MLAAttention, "_q_proj_and_k_up_proj", q_proj)
    monkeypatch.setattr(
        attention_mla.MLAAttention,
        "_dsa_row_parallel_prefill",
        lambda *a: pytest.fail("row-parallel without a plan"),
    )
    with pytest.raises(_Stop):
        impl.forward_impl(*_inputs(40))


def test_a_sparse_prefill_with_a_plan_runs_row_parallel(monkeypatch):
    plan = _plan(world=W)
    impl, md, kv_cache = _mla(monkeypatch, plan)
    seen = []
    monkeypatch.setattr(
        attention_mla.MLAAttention,
        "_dsa_row_parallel_prefill",
        lambda self, *a: seen.append(a) or "out",
    )
    q, k_nope, k_rope = _inputs(plan.num_tokens)
    assert impl.forward_impl(q, k_nope, k_rope) == "out"
    ((q_, kn, kr, q_scale, cache, md_, plan_),) = seen
    assert q_ is q and kn is k_nope and kr is k_rope and q_scale is None
    assert cache is kv_cache and md_ is md and plan_ is plan


def test_a_layer_that_did_not_opt_in_refuses_the_step(monkeypatch):
    impl, _, _ = _mla(monkeypatch, _plan(world=W), opted_in=False)
    with pytest.raises(RuntimeError):
        impl.forward_impl(*_inputs(40))


@pytest.mark.parametrize("opted_in", [False, True])
def test_a_dummy_prefill_holds_the_gather_only_when_opted_in(monkeypatch, opted_in):
    impl, _, _ = _mla(monkeypatch, None, dummy=True, opted_in=opted_in, tokens=40)
    held = []
    monkeypatch.setattr(
        attention_mla.MLAAttention,
        "_hold_dsa_row_parallel_scratch",
        lambda self, device: held.append(device),
    )
    monkeypatch.setattr(attention_mla, "_mla_output_width", lambda *a: 12)
    monkeypatch.setattr(
        attention_mla,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            hf_config=SimpleNamespace(hidden_size=12), torch_dtype=torch.bfloat16
        ),
    )
    impl.forward_impl(*_inputs(40))
    assert held == ([torch.device("cpu")] if opted_in else [])


def test_the_hold_is_the_scratch_at_the_cap(monkeypatch):
    impl = _impl()
    object.__setattr__(impl, "kv_lora_rank", 512)
    object.__setattr__(impl, "qk_rope_head_dim", 64)
    monkeypatch.setenv("ATOM_DSA_ROW_PARALLEL_PREFILL_MAX_KV_ROWS", "4096")
    sizes = []
    empty = torch.empty

    def record(*shape, **kw):
        sizes.append((shape, kw.get("dtype")))
        return empty(0)

    monkeypatch.setattr(torch, "empty", record)
    impl._hold_dsa_row_parallel_scratch("cpu")
    assert sizes == [((rp.scratch_bytes(4096, 8, 576),), torch.uint8)]


def test_a_row_parallel_layer_runs_its_rows(monkeypatch):
    """The call sequence, each kernel on this rank's rows: the KV written for
    every row, q projected and absorbed for the plan's rows, the whole top-k
    attended over the gathered KV with no LSE, V projected head-major, and the
    rows handed back to heads."""
    plan = _plan(world=W)
    n = plan.rows
    impl, md, kv_cache = _mla(monkeypatch, plan)
    calls = []
    q_nope = torch.zeros(n, H, 6, dtype=torch.bfloat16)
    q_rope = torch.zeros(n, H, R, dtype=torch.bfloat16)

    def q_proj(self, x, x_scale=None, group=False, k_up_proj=True):
        calls.append(("q_proj", x, x_scale, group, k_up_proj))
        return q_nope, q_rope

    monkeypatch.setattr(attention_mla.MLAAttention, "_q_proj_and_k_up_proj", q_proj)
    monkeypatch.setattr(rp, "write_kv", lambda *a: calls.append(("write_kv", a)))
    gathered = torch.zeros(80, 1, L + R, dtype=torch.float8_e4m3fn)
    monkeypatch.setattr(
        rp, "gather_kv", lambda *a: calls.append(("gather", a)) or gathered
    )
    monkeypatch.setattr(
        attention_mla, "mla_q_tail_and_cache", lambda *a: calls.append(("q_tail", a))
    )
    monkeypatch.setattr(
        attention_mla, "mla_absorb_q_pack", lambda *a: calls.append(("absorb", a))
    )
    monkeypatch.setattr(
        attention_mla,
        "mla_decode_fwd",
        lambda *a, **k: calls.append(("mla", a, k)) or (None, None),
    )
    monkeypatch.setattr(
        attention_mla,
        "_aiter_triton_bf16_bmm",
        lambda x, w, YQ: calls.append(("v_up", x, w, YQ)),
    )
    out = torch.zeros(plan.num_tokens, (H // W) * V)
    monkeypatch.setattr(rp, "rows_to_heads", lambda *a: calls.append(("a2a", a)) or out)
    q, k_nope, k_rope = _inputs(plan.num_tokens)
    assert impl.forward_impl(q, k_nope, k_rope) is out

    names = [c[0] for c in calls]
    assert names == [
        "write_kv",
        "q_proj",
        "q_tail",
        "absorb",
        "gather",
        "mla",
        "v_up",
        "a2a",
    ]
    by = {c[0]: c for c in calls}
    _, (kv_c, k_pe, cache, slots, k_scale) = by["write_kv"]
    assert kv_c is k_nope and k_pe is k_rope and slots is md.slot_mapping
    assert cache.shape == (64, 1, L + R) and k_scale is impl._k_scale
    _, x, x_scale, group, k_up = by["q_proj"]
    assert (
        torch.equal(x, q[plan.first : plan.end])
        and x.data_ptr() == q[plan.first].data_ptr()
    )
    assert x_scale is None and group is True and k_up is False
    _, (qr, kvc, kpe, _cache2, q_out, slots2, _ks, _qs) = by["q_tail"]
    assert (
        qr is q_rope and kvc.shape[0] == kpe.shape[0] == n and slots2 is plan.no_slots
    )
    assert q_out.shape == (n, H, L + R) and q_out.dtype == md.dtype_q
    _, (qn, w_k, q_out2, _qs2) = by["absorb"]
    assert w_k == "W_K_qrep" and q_out2 is q_out and qn.shape == (H, n, 6)
    _, (group_, cache3, kv_plan) = by["gather"]
    assert group_ == "group" and cache3 is kv_cache and kv_plan is plan.kv
    _, args, kw = by["mla"]
    kq, kv, o, cu, indptr, indices, last, max_q = args
    assert kq is q_out and kv.shape == (80, 1, 1, L + R) and o.shape == (n, H, L)
    assert (
        cu is plan.cu_seqlens_q
        and indptr is plan.kv_indptr
        and last is plan.last_page_lens
    )
    assert indices.data_ptr() == md.sparse_kv_indices[plan.kv_base :].data_ptr()
    assert max_q == 1 and kw["page_size"] == 1 and kw["return_lse"] is False
    assert kw["work_indptr"] is plan.work["work_indptr"]
    assert kw["sm_scale"] == 0.5 and kw["q_scale"] is impl._q_scale
    _, x_v, w_v, yq = by["v_up"]
    assert x_v.shape == (H, n, L) and w_v is impl.W_V_dcp and yq.shape == (H, n, V)
    _, (projected, rows, group2) = by["a2a"]
    assert projected.shape == (H, plan.stride, V) and rows == plan.num_tokens
    assert yq.data_ptr() == projected.data_ptr() and group2 == "group"


def test_a_row_parallel_layer_refuses_a_scaled_q(monkeypatch):
    plan = _plan(world=W)
    impl, _, _ = _mla(monkeypatch, plan)
    q, k_nope, k_rope = _inputs(plan.num_tokens)
    with pytest.raises(RuntimeError):
        impl.forward_impl(q, k_nope, k_rope, q_scale=torch.ones(plan.num_tokens, 1))


# ───────────────────────────────────────────── the kernels (GPU) ──


@gpu
@pytest.mark.parametrize("world,block", [(8, 64), (4, 1)])
def test_the_index_kernel_maps_as_the_reference(world, block):
    context_lens = [300, 129, 4097]
    tables, _ = _requests(context_lens, world, block)
    plan = rp.gathered_kv_plan(np.array(context_lens), tables, world, block, "cuda")
    g = torch.Generator().manual_seed(0)
    topk_width, rows = 256, 64
    batch_ids = torch.randint(
        0, len(context_lens), (rows,), generator=g, dtype=torch.int32
    )
    topk = torch.full((rows, topk_width), -1, dtype=torch.int32)
    for t in range(rows):
        n = context_lens[int(batch_ids[t])]
        k = int(torch.randint(1, min(n, topk_width) + 1, (1,), generator=g))
        topk[t, :k] = torch.randperm(n, generator=g)[:k].int()
        if t % 7 == 0:
            topk[t, k // 2] = -1  # an invalid candidate inside the count
    counts = (topk >= 0).sum(1) + (torch.arange(rows) % 7 == 0).int()
    counts = torch.minimum(counts, torch.full_like(counts, topk_width))
    indptr = torch.zeros(rows + 1, dtype=torch.int32)
    indptr[1:] = counts.cumsum(0)
    want = ref_index(
        indptr.numpy(),
        batch_ids.numpy(),
        topk.numpy(),
        plan.shard_base.cpu().numpy(),
        plan.rows_pad,
        world,
        np.full(int(indptr[-1]) + 5, -7, dtype=np.int64),
    )
    for first, end in ((0, rows), (17, 40)):
        out = torch.full((int(indptr[-1]) + 5,), -7, dtype=torch.int32, device="cuda")
        rp.gathered_kv_index(
            indptr[first : end + 1].cuda(),
            batch_ids[first:end].cuda(),
            topk[first:end].cuda(),
            plan,
            world,
            out,
            topk_width,
        )
        lo, hi = int(indptr[first]), int(indptr[end])
        assert out[lo:hi].cpu().tolist() == want[lo:hi].tolist()
        assert (out[:lo] == -7).all() and (out[hi:] == -7).all()


@gpu
def test_the_kv_write_writes_the_q_tail_kernels_codes():
    from aiter import dtypes

    from atom.model_ops.triton_mla_q_tail_cache import mla_q_tail_and_cache

    g = torch.Generator(device="cuda").manual_seed(0)
    tokens, heads, blocks, block = 300, 4, 50, 8
    kv_c = torch.randn(tokens, 512, device="cuda", generator=g).bfloat16() * 30
    k_pe = torch.randn(tokens, 64, device="cuda", generator=g).bfloat16() * 30
    slots = torch.randperm(blocks * block, device="cuda", generator=g)[:tokens]
    slots[::3] = -1
    for scale in (torch.tensor(0.7), torch.tensor(0.7, device="cuda")):
        want = torch.zeros(blocks, block, 576, dtype=dtypes.fp8, device="cuda")
        got = torch.zeros_like(want)
        q_out = torch.empty(tokens, heads, 576, dtype=dtypes.fp8, device="cuda")
        q_pe = torch.randn(tokens, heads, 64, device="cuda", generator=g).bfloat16()
        mla_q_tail_and_cache(q_pe, kv_c, k_pe, want, q_out, slots, scale, scale)
        rp.write_kv(kv_c, k_pe, got, slots, scale)
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


# ─────────────────── the front on its rows (ATOM_DSA_ROW_PARALLEL_PREFILL_FRONT) ──


def ref_index_new(indptr, batch_ids, topk, plan, world, out):
    """The index kernel's mapping with new rows, row by row."""
    cached, first, count = plan.new_meta.cpu().numpy().reshape(3, -1)
    shard_base = plan.shard_base.cpu().numpy()
    stride, block_rows = plan.new_stride, plan.block_rows
    for t in range(len(batch_ids)):
        req, start = int(batch_ids[t]), int(indptr[t])
        for c in range(int(indptr[t + 1]) - start):
            p = int(topk[t, c])
            if p >= cached[req]:
                j = int(first[req]) + p - int(cached[req])
                ok = p - cached[req] < count[req]
                row = (j // stride) * block_rows + plan.rows_pad + j % stride
            else:
                base, end = int(shard_base[req]), int(shard_base[req + 1])
                local = p // world
                ok = p >= 0 and base + local < end
                row = (p % world) * block_rows + base + local
            out[start + c] = row if ok else 0
    return out


def _front_plan(cached, new, world, block, device="cpu"):
    tables, n_blocks = _requests([c + n for c, n in zip(cached, new)], world, block)
    stride = -(-sum(new) // world)
    kv = rp.gathered_kv_plan(np.array(cached), tables, world, block, device)
    return rp.with_new_rows(kv, cached, new, stride, device), tables, n_blocks


def _gathered_front(cached, new, tables, world, block, n_blocks, plan):
    """Each rank's block of `gather_local_kv`, by global ids (request,
    position): its cache's rows of the cached positions, then the new
    tokens of its rows. Returns it and the step's new tokens in order."""
    caches = [np.full((n_blocks * block, 2), -1, dtype=np.int64) for _ in range(world)]
    for i, n in enumerate(cached):
        for p in range(n):
            owner, slot = owner_slot(tables[i], p, world, block)
            caches[owner][slot] = (i, p)
    tokens = [(i, cached[i] + t) for i in range(len(new)) for t in range(new[i])]
    slots, stride = plan.slot_ids.numpy(), plan.new_stride
    blocks = []
    for r in range(world):
        rows = np.full((stride, 2), -1, dtype=np.int64)
        mine = tokens[r * stride : (r + 1) * stride]
        if mine:
            rows[: len(mine)] = mine
        blocks.append(np.concatenate([caches[r][slots], rows]))
    return np.concatenate(blocks), tokens


def _candidates(cached, new, topk_width=16, seed=1):
    """Three query rows a request, each a top-k of the positions it sees."""
    rng = np.random.default_rng(seed)
    batch_ids, rows = [], []
    for i, (c, n) in enumerate(zip(cached, new)):
        for _ in range(3):
            visible = c + int(rng.integers(1, n + 1))
            k = min(visible, topk_width)
            row = np.full(topk_width, -1, dtype=np.int32)
            row[:k] = rng.choice(visible, size=k, replace=False)
            batch_ids.append(i)
            rows.append(row)
    topk = np.stack(rows)
    counts = (topk >= 0).sum(1)
    return batch_ids, topk, np.concatenate([[0], np.cumsum(counts)]), counts


def test_the_front_sub_switch_is_on_by_default(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_DSA_ROW_PARALLEL_PREFILL_FRONT", raising=False)
    assert envs.ATOM_DSA_ROW_PARALLEL_PREFILL_FRONT is True
    assert aiter_mla.AiterMLAMetadataBuilder._dsa_row_parallel_front is False


@pytest.mark.parametrize("world,block", [(8, 64), (4, 4), (2, 1)])
@pytest.mark.parametrize(
    "cached,new", [([0], [300]), ([129, 0, 70], [5, 300, 1]), ([64 * 8 * 3], [1000])]
)
def test_with_new_rows_every_candidate_reads_its_position(world, block, cached, new):
    plan, tables, n_blocks = _front_plan(cached, new, world, block)
    stride = plan.new_stride
    assert plan.block_rows == plan.rows_pad + stride
    assert plan.rows_pad * world == rp.gathered_kv_rows(cached, world, block)
    gathered, tokens = _gathered_front(
        cached, new, tables, world, block, n_blocks, plan
    )
    # `write_new_kv` reads token j's KV from the row its rank wrote it to.
    for j, token in enumerate(tokens):
        row = (j // stride) * plan.block_rows + plan.rows_pad + j % stride
        assert tuple(gathered[row]) == token
    batch_ids, topk, indptr, counts = _candidates(cached, new)
    out = ref_index_new(
        indptr, batch_ids, topk, plan, world, np.full(indptr[-1], -7, dtype=np.int64)
    )
    for t, i in enumerate(batch_ids):
        for c in range(counts[t]):
            assert tuple(gathered[out[indptr[t] + c]]) == (i, int(topk[t, c]))


def test_new_rows_take_each_requests_cached_first_and_new_tokens():
    plan, _, _ = _front_plan([129, 0, 70], [5, 300, 1], 8, 64)
    assert plan.new_meta.view(3, -1).tolist() == [
        [129, 0, 70],
        [0, 5, 305],
        [5, 300, 1],
    ]
    assert plan.new_stride == 39 and plan.rows_pad == 128


@pytest.mark.parametrize("stride,new", [(0, [5]), (4, [5, 6])])
def test_new_rows_refuse_a_bad_layout(stride, new):
    kv = rp.gathered_kv_plan(np.array([0]), [[0]], 8, 64, "cpu")
    with pytest.raises(ValueError):
        rp.with_new_rows(kv, [0], new, stride, "cpu")


def test_the_local_block_is_the_shard_then_room_for_its_rows():
    cache = torch.arange(40 * 4, dtype=torch.uint8).view(40, 1, 4)
    cache = cache.view(torch.float8_e4m3fn)
    plan = rp.GatheredKV(
        slot_ids=torch.tensor([7, 3, 30], dtype=torch.int32),
        shard_base=torch.tensor([0, 3], dtype=torch.int32),
        rows_pad=3,
        new_stride=2,
        new_meta=torch.zeros(3, dtype=torch.int32),
    )
    local = rp.local_kv(cache, plan)
    assert local.shape == (5, 4) and local.dtype == cache.dtype
    want = cache.view(torch.uint8)[[7, 3, 30], 0]
    assert torch.equal(local[:3].view(torch.uint8), want)


@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("ctx,tokens", [(0, 16384), (114688, 16384), (300, 8195)])
def test_a_front_step_carries_its_new_rows_in_the_gather(
    monkeypatch, rank, ctx, tokens
):
    b, _calls, _plans = _builder(monkeypatch, rank=rank)
    b._dsa_row_parallel_front = True
    batch = _batch([ctx + tokens], [tokens])
    md = _metadata(tokens)
    _finish(b, batch, _staged(batch), md)
    plan = md.dsa_row_parallel
    stride = -(-tokens // WORLD)
    assert plan.front and plan.kv.new_stride == stride == plan.stride
    assert plan.kv.rows_pad * WORLD == rp.gathered_kv_rows([ctx], WORLD, BLOCK)
    assert plan.kv.new_meta.tolist() == [ctx, 0, tokens]
    pad = plan.kv.rows_pad
    assert plan.new_slots.tolist() == list(range(pad, pad + plan.rows))
    assert plan.no_slots.tolist() == [-1] * plan.rows


@pytest.mark.parametrize("front", [False, True])
def test_the_cap_counts_the_rows_each_layout_gathers(monkeypatch, front):
    # 8195 new tokens: 8 x 1025 rows a rank with the front, 8 x 17 x 64 without.
    b, _calls, _plans = _builder(monkeypatch, cap=8 * 1025)
    b._dsa_row_parallel_front = front
    batch = _batch([8195], [8195])
    md = _metadata(8195)
    _finish(b, batch, _staged(batch), md)
    assert (md.dsa_row_parallel is not None) is front


def test_a_front_layer_carries_its_rows_kv_in_the_gather(monkeypatch):
    """On a front step a layer reads only its rows of the KV: their codes go
    after its shard of the gathered KV, at `new_slots`, with no KV write for
    every row; each rank copies its owned slots out of the gather."""
    from dataclasses import replace

    plan = _plan(world=W)
    n = plan.rows
    kv_plan = replace(
        plan.kv, new_stride=plan.stride, new_meta=torch.zeros(3, dtype=torch.int32)
    )
    plan = replace(
        plan, kv=kv_plan, new_slots=torch.arange(8, 8 + n, dtype=torch.int64)
    )
    impl, md, kv_cache = _mla(monkeypatch, plan)
    calls = []
    q_nope = torch.zeros(n, H, 6, dtype=torch.bfloat16)
    q_rope = torch.zeros(n, H, R, dtype=torch.bfloat16)
    monkeypatch.setattr(
        attention_mla.MLAAttention,
        "_q_proj_and_k_up_proj",
        lambda self, x, **k: calls.append(("q_proj", x)) or (q_nope, q_rope),
    )
    local = torch.zeros(kv_plan.block_rows, L + R, dtype=torch.float8_e4m3fn)
    gathered = torch.zeros(W * kv_plan.block_rows, 1, L + R, dtype=local.dtype)
    monkeypatch.setattr(rp, "write_kv", lambda *a: pytest.fail("every row's KV"))
    monkeypatch.setattr(rp, "gather_kv", lambda *a: pytest.fail("the cache gather"))
    monkeypatch.setattr(rp, "local_kv", lambda *a: calls.append(("local", a)) or local)
    monkeypatch.setattr(
        rp, "gather_local_kv", lambda *a: calls.append(("gather", a)) or gathered
    )
    monkeypatch.setattr(rp, "write_new_kv", lambda *a: calls.append(("new_kv", a)))
    monkeypatch.setattr(
        attention_mla, "mla_q_tail_and_cache", lambda *a: calls.append(("q_tail", a))
    )
    monkeypatch.setattr(
        attention_mla, "mla_absorb_q_pack", lambda *a: calls.append(("absorb", a))
    )
    monkeypatch.setattr(
        attention_mla,
        "mla_decode_fwd",
        lambda *a, **k: calls.append(("mla", a, k)) or (None, None),
    )
    monkeypatch.setattr(attention_mla, "_aiter_triton_bf16_bmm", lambda *a, **k: None)
    out = torch.zeros(plan.num_tokens, (H // W) * V)
    monkeypatch.setattr(rp, "rows_to_heads", lambda *a: calls.append(("a2a", a)) or out)
    q, k_nope, k_rope = _inputs(plan.num_tokens)
    assert impl.forward_impl(q, k_nope, k_rope) is out

    names = [c[0] for c in calls]
    assert names == [
        "local",
        "q_proj",
        "q_tail",
        "absorb",
        "gather",
        "new_kv",
        "mla",
        "a2a",
    ]
    by = {c[0]: c for c in calls}
    assert by["local"][1] == (kv_cache, kv_plan)
    _, (_qr, kvc, kpe, tail_cache, _q_out, slots, _ks, _qs) = by["q_tail"]
    own = slice(plan.first, plan.end)
    assert kvc.data_ptr() == k_nope[own].data_ptr() and kvc.shape[0] == n
    assert kpe.data_ptr() == k_rope[own].data_ptr()
    assert tail_cache.shape == (kv_plan.block_rows, 1, L + R)
    assert tail_cache.data_ptr() == local.data_ptr() and slots is plan.new_slots
    assert by["gather"][1] == ("group", local)
    kv, cache, slot_mapping, plan_kv = by["new_kv"][1]
    assert kv is gathered and cache is kv_cache and plan_kv is kv_plan
    assert slot_mapping is md.slot_mapping
    assert by["mla"][1][1].shape == (W * kv_plan.block_rows, 1, 1, L + R)


@gpu
@pytest.mark.parametrize("world,block", [(8, 64), (4, 1)])
def test_the_index_kernel_maps_new_rows_as_the_reference(world, block):
    cached, new = [300, 0, 4097], [129, 700, 3]
    plan, _, _ = _front_plan(cached, new, world, block, "cuda")
    batch_ids, topk, indptr, _counts = _candidates(cached, new, topk_width=256)
    want = ref_index_new(
        indptr,
        batch_ids,
        topk,
        plan,
        world,
        np.full(int(indptr[-1]) + 5, -7, dtype=np.int64),
    )
    rows = len(batch_ids)
    out = torch.full((int(indptr[-1]) + 5,), -7, dtype=torch.int32, device="cuda")
    rp.gathered_kv_index(
        torch.from_numpy(indptr.astype(np.int32)).cuda(),
        torch.tensor(batch_ids, dtype=torch.int32, device="cuda"),
        torch.from_numpy(topk).cuda(),
        plan,
        world,
        out,
        256,
    )
    hi = int(indptr[rows])
    assert out[:hi].cpu().tolist() == want[:hi].tolist()
    assert (out[hi:] == -7).all()


@gpu
def test_new_kv_lands_at_the_owned_slots_only():
    from aiter import dtypes

    g = torch.Generator(device="cuda").manual_seed(0)
    world, rows_pad, stride, tokens = 4, 5, 7, 26
    plan = rp.GatheredKV(
        slot_ids=torch.zeros(rows_pad, dtype=torch.int32, device="cuda"),
        shard_base=torch.tensor([0, rows_pad], dtype=torch.int32, device="cuda"),
        rows_pad=rows_pad,
        new_stride=stride,
        new_meta=torch.zeros(3, dtype=torch.int32, device="cuda"),
    )
    codes = torch.randint(
        0, 256, (world * plan.block_rows, 1, 576), generator=g, device="cuda"
    )
    gathered = codes.to(torch.uint8).view(dtypes.fp8)
    slots = torch.randperm(64, device="cuda", generator=g)[:tokens]
    slots[::3] = -1
    cache = torch.zeros(64, 1, 576, dtype=dtypes.fp8, device="cuda")
    rp.write_new_kv(gathered, cache, slots, plan)
    want = torch.zeros(64, 576, dtype=torch.uint8)
    src = gathered.view(torch.uint8).view(-1, 576).cpu()
    for j, slot in enumerate(slots.tolist()):
        if slot >= 0:
            want[slot] = src[(j // stride) * plan.block_rows + rows_pad + j % stride]
    assert torch.equal(cache.view(torch.uint8).view(64, 576).cpu(), want)
