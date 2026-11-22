"""Execute the source-bound builder/staging bodies without importing GPU modules."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from atom.distributed import group_agreement
from atom.model_ops.attentions.mla_kv_pool import MlaKvPool
from atom.model_ops.indexer.layout import fp4_scale_byte
from atom.model_ops.sparse_indexer_fp4 import fp4_index_scale_rows
from atom.utils import envs
from atom.utils.forward_context import Context, ForwardContext

ROOT = Path(__file__).resolve().parents[1]
BUILDER = "atom/model_ops/attentions/aiter_mla.py"
STAGING = "atom/model_ops/indexer/indexer_forward.py"


def source_function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(nodes) == 1
    exec(  # noqa: S102 - execute the unchanged production body at controlled boundaries
        compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"),
        scope,
    )
    return scope[name]


def functions(group=None):
    scope = {
        "torch": torch,
        "np": np,
        "logging": logging,
        "MFMA": "mfma",
        "PRESHUFFLE": "preshuffle",
        "fp4_scale_byte": fp4_scale_byte,
        "fp4_index_scale_rows": fp4_index_scale_rows,
        "upload_numpy": lambda a, device: torch.from_numpy(a.copy()).to(device),
        "get_dcp_group": lambda: group,
        "get_forward_context": lambda: ForwardContext(
            context=Context(
                positions=torch.empty(0, dtype=torch.int64), is_prefill=True
            ),
            in_hipgraph=False,
        ),
        "_FP4_COMBINED_REPORTED": False,
    }
    return (
        source_function(BUILDER, "_build_dcp_indexer_fp4_prefill_meta", scope),
        source_function(STAGING, "_dcp_stage_indexer_fp4_prefill", scope),
    )


def metadata(build, lengths, tables, world, interleave, combined, builder=None):
    lens = np.asarray(lengths, dtype=np.int64)
    local = ((lens + world * interleave - 1) // (world * interleave)) * interleave
    cu = np.r_[0, np.cumsum(local)]
    global_cu = np.r_[0, np.cumsum(lens)]
    builder = builder or NS(
        model_runner=NS(block_size=64),
        device="cpu",
        block_table_cols=tables.shape[1],
        _fp4_prefill_combined=combined,
    )
    result = NS()
    build(builder, result, len(lens), local, cu, global_cu, lens, tables)
    gathered = []
    sources = []
    destination = []
    start_page = 0
    for request, length in enumerate(lengths):
        for token in range(length):
            rank = (token // interleave) % world
            offset = token // (world * interleave) * interleave + token % interleave
            slot = int(tables[request, offset // 64]) * 64 + offset % 64
            sources.append((rank, slot))
            gathered.append(rank * int(cu[-1]) + int(cu[request]) + offset)
            destination.append(start_page * 64 + token)
        start_page += (length + 63) // 64
    result.dcp_indexer_gather_index = torch.tensor(gathered, dtype=torch.int32)
    slots = [
        int(tables[r, j // 64]) * 64 + j % 64
        for r in range(len(lengths))
        for j in range(int(local[r]))
    ]
    return result, slots, sources, destination


def decode_rows(data, exponents):
    code = torch.stack((data & 15, data >> 4), -1).reshape(*data.shape[:-1], 128)
    magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[
        (code & 7).long()
    ]
    sign = torch.where(code & 8 != 0, -1.0, 1.0)
    scale = torch.pow(2.0, exponents.float() - 127).repeat_interleave(32, -1)
    return magnitude * sign * scale


@pytest.mark.parametrize("world,interleave", [(2, 1), (8, 1), (8, 4)])
@pytest.mark.parametrize(
    "lengths", [(1,), (63,), (64,), (65,), (129, 257), (1, 64, 131)]
)
def test_real_builder_to_staging_bytes_scales_padding_and_layer_refresh(
    world, interleave, lengths
):
    torch.manual_seed(29)
    local = [
        ((n + world * interleave - 1) // (world * interleave)) * interleave
        for n in lengths
    ]
    blocks = [(n + 63) // 64 for n in local]
    pages = sum(blocks) + 3
    tables = np.full((len(lengths), max(blocks)), -999, dtype=np.int32)
    permutation = torch.randperm(pages).tolist()
    cursor = 0
    for request, count in enumerate(blocks):
        tables[request, :count] = permutation[cursor : cursor + count]
        cursor += count
    pools = []
    for _ in range(world):
        pool = MlaKvPool(
            layers=1,
            block_size=64,
            entry_dim=576,
            kv_dtype=torch.bfloat16,
            index_layers=1,
            index_rows_per_block=64,
            index_head_dim=128,
            index_fp4=True,
            index_layout="mfma",
        )
        pool.allocate(pages, "cpu")
        pools.append(pool)
    for layer in range(2):
        natural = torch.randint(0, 256, (world, pages, 64, 64), dtype=torch.uint8)
        exponents = torch.randint(120, 135, (world, pages, 64, 4), dtype=torch.uint8)
        for rank, pool in enumerate(pools):
            pool.layer("index", 0).copy_(
                natural[rank].reshape(pages, 64, 4, 16).permute(0, 2, 1, 3)
            )
            physical_scale = (
                exponents[rank].reshape(pages, 2, 32, 2, 2).permute(0, 4, 2, 3, 1)
            )
            pool.layer("index_scale", 0).copy_(physical_scale.reshape(pages, 2, 32, 4))
        for combined in (False, True):
            for rank, pool in enumerate(pools):
                calls = []
                group = NS(world_size=world)
                build, stage = functions(group)
                meta, slots, sources, destination = metadata(
                    build, lengths, tables, world, interleave, combined
                )
                all_data = [
                    natural[r].reshape(-1, 64)[slots].reshape(-1, 4, 16)
                    for r in range(world)
                ]
                all_scales = [exponents[r].reshape(-1, 4)[slots] for r in range(world)]

                def gather(
                    value,
                    dim,
                    use_custom=False,
                    *,
                    combined=combined,
                    all_data=all_data,
                    all_scales=all_scales,
                    rank=rank,
                    calls=calls,
                ):
                    assert (
                        dim == 0 and use_custom is False and value.dtype == torch.uint8
                    )
                    if combined:
                        payloads = [
                            torch.cat((d.flatten(1), s), 1)
                            for d, s in zip(all_data, all_scales, strict=True)
                        ]
                        assert value.is_contiguous() and value.shape[1] == 68
                    else:
                        payloads = all_data if not calls else all_scales
                    assert torch.equal(value, payloads[rank])
                    calls.append(value.shape)
                    return torch.cat(payloads)

                group.all_gather = gather
                data, scale = pool.layer("index", 0), pool.layer("index_scale", 0)
                original = (data.clone(), scale.clone())
                staged, staged_scale = stage(data, scale, meta, "mfma")
                assert len(calls) == (1 if combined else 2)
                assert torch.equal(data, original[0]) and torch.equal(
                    scale, original[1]
                )
                n_pages = meta.dcp_indexer_fp4_stage_pages
                logical = staged.permute(0, 2, 1, 3).reshape(-1, 64)
                logical_scale = (
                    staged_scale.reshape(n_pages, 2, 32, 2, 2)
                    .permute(0, 4, 2, 3, 1)
                    .reshape(-1, 4)
                )
                expected, expected_scale = torch.zeros_like(logical), torch.zeros_like(
                    logical_scale
                )
                for dst, (peer, slot) in zip(destination, sources, strict=True):
                    expected[dst] = natural[peer].reshape(-1, 64)[slot]
                    expected_scale[dst] = exponents[peer].reshape(-1, 4)[slot]
                assert torch.equal(logical, expected), (layer, rank, combined)
                assert torch.equal(logical_scale, expected_scale), (
                    layer,
                    rank,
                    combined,
                )
                assert torch.equal(
                    decode_rows(logical, logical_scale),
                    decode_rows(expected, expected_scale),
                )


def test_forward_refresh_and_large_scale_offsets_do_not_overflow():
    build, _ = functions()
    page = (2**29 // 64) + 3
    tables = np.array([[page]], dtype=np.int32)
    builder = NS(
        model_runner=NS(block_size=64),
        device="cpu",
        block_table_cols=1,
        _fp4_prefill_combined=True,
    )
    first, _, _, _ = metadata(build, (65,), tables, 8, 1, True, builder)
    before = first.dcp_indexer_fp4_read_scale_at.clone()
    second, _, _, _ = metadata(build, (129,), tables + 5, 8, 1, True, builder)
    assert first.dcp_indexer_fp4_read_page.dtype == torch.int64
    assert first.dcp_indexer_fp4_read_row.dtype == torch.int64
    assert before.dtype == torch.int64 and before.min() > 2**31
    # Natural [page,row,chunk] indexing, independent of fp4_scale_byte.
    for j, row in enumerate(before):
        expected = [
            page * 256 + (c % 2) * 128 + (j % 32) * 4 + (c // 2) * 2 + j // 32
            for c in range(4)
        ]
        assert row.tolist() == expected
    assert second.dcp_indexer_fp4_read_scale_at.shape[0] == 17
    assert torch.equal(first.dcp_indexer_fp4_read_scale_at, before)
    assert (
        second.dcp_indexer_fp4_read_scale_at.data_ptr()
        != first.dcp_indexer_fp4_read_scale_at.data_ptr()
    )
    assert second.dcp_indexer_fp4_read_scale_at[0, 0] == (page + 5) * 256


@pytest.mark.parametrize(
    "requested,eligible,peer_ok",
    [(False, True, True), (True, False, True), (True, True, False), (True, True, True)],
)
def test_off_unsupported_and_enabled_ranks_all_participate_in_agreement(
    monkeypatch, requested, eligible, peer_ok
):
    group = object()
    calls = []

    def reduce(value, op, group):
        assert value.device.type == "cpu" and value.dtype == torch.int64
        assert op == torch.distributed.ReduceOp.MAX
        calls.append(group)
        value[0] = max(value[0].item(), int(not peer_ok))

    monkeypatch.setattr(group_agreement.dist, "all_reduce", reduce)
    scope = {
        "MFMA": "mfma",
        "envs": NS(ATOM_DCP_FP4_PREFILL_COMBINED=requested),
        "group_agree": group_agreement.group_agree,
        "get_dcp_group": lambda: NS(cpu_group=group),
    }
    policy = source_function(BUILDER, "_agree_fp4_prefill_combined", scope)
    runner = NS(
        block_size=64,
        config=NS(
            index_cache_layout="mfma" if eligible else "preshuffle",
            hf_config=NS(index_head_dim=128),
        ),
    )
    assert policy(runner, True, 8) == (requested and eligible and peer_ok)
    assert calls == [group]
    calls.clear()
    assert not policy(runner, True, 1) and not calls


def test_flag_is_default_off_and_explicit_one_enables(monkeypatch):
    monkeypatch.delenv("ATOM_DCP_FP4_PREFILL_COMBINED", raising=False)
    assert not envs.ATOM_DCP_FP4_PREFILL_COMBINED
    monkeypatch.setenv("ATOM_DCP_FP4_PREFILL_COMBINED", "0")
    assert not envs.ATOM_DCP_FP4_PREFILL_COMBINED
    monkeypatch.setenv("ATOM_DCP_FP4_PREFILL_COMBINED", "1")
    assert envs.ATOM_DCP_FP4_PREFILL_COMBINED


def test_one_forward_metadata_reads_each_layer_and_keeps_prior_outputs():
    calls = []

    def gather(value, dim, use_custom):
        assert dim == 0 and not use_custom
        calls.append(value.shape)
        return torch.cat((value, value))

    build, stage = functions(NS(world_size=2, all_gather=gather))
    meta, _, _, _ = metadata(build, (65,), np.array([[1]], dtype=np.int32), 2, 1, True)
    data = torch.randint(0, 256, (3, 4, 64, 16), dtype=torch.uint8)
    scale = torch.randint(120, 135, (3, 2, 32, 4), dtype=torch.uint8)
    first = stage(data, scale, meta, "mfma")
    saved = tuple(value.clone() for value in first)
    data.bitwise_xor_(0x5A)
    scale.add_(1)
    second = stage(data, scale, meta, "mfma")
    assert len(calls) == 2
    for before, snapshot, after in zip(first, saved, second, strict=True):
        assert torch.equal(before, snapshot)
        assert before.data_ptr() != after.data_ptr()
        assert not torch.equal(after, before)


def test_real_cache_initializer_and_custom_op_scale_contract():
    import __future__

    tree = ast.parse((ROOT / "atom/models/deepseek_v2.py").read_text())
    cache = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV32IndexerCache"
    )
    scope = {
        "torch": torch,
        "nn": torch.nn,
        # Native-mode boundary: the plugin decorator leaves this class unchanged.
        "DeepseekV32IndexerCacheDecoratorForPluginMode": lambda cls: cls,
    }
    exec(  # noqa: S102 - source-bound real cache class, native decorator boundary
        compile(
            ast.Module(body=[cache], type_ignores=[]),
            "deepseek_v2.py",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    instance = scope[cache.name](132, torch.uint8, "indexer.k_cache", "fp8")
    assert isinstance(instance.kv_cache_scale, torch.Tensor)
    assert (
        instance.kv_cache_scale.dtype == torch.uint8
        and instance.kv_cache_scale.numel() == 0
    )
    rebound = torch.empty((2, 2, 32, 4), dtype=torch.uint8)
    instance.kv_cache_scale = rebound
    assert instance.kv_cache_scale is rebound
    source = ast.parse((ROOT / STAGING).read_text())
    function = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_indexer"
    )
    assert "kv_cache_scale" in [arg.arg for arg in function.args.args]
    registration = next(
        n
        for n in ast.walk(source)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "direct_register_custom_op"
    )
    options = {keyword.arg: keyword.value for keyword in registration.keywords}
    assert options["op_func"].id == "sparse_attn_indexer"
    assert set(ast.literal_eval(options["mutates_args"])) == {
        "kv_cache",
        "kv_cache_scale",
        "sparse_kv_indices_buffer",
        "dcp_sparse_kv_indptr_buffer",
        "dcp_owned_counts_buffer",
    }
