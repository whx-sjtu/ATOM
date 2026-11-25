# SPDX-License-Identifier: MIT
"""Kimi row-sharded residual and MoE tail on CPU-simulated TP ranks."""

import ast
import gc
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from torch import nn

kimi = module_dispatch_ops = kimi_k3_ops = sequence_parallel = ubatching = None
attention_residual = None


@pytest.fixture(scope="module", autouse=True)
def imported_modules():
    global kimi, module_dispatch_ops, kimi_k3_ops, sequence_parallel, ubatching
    global attention_residual
    with atom_without_aiter() as load:
        kimi = load("atom.models.kimi_k3.model")
        module_dispatch_ops = load("atom.model_ops.module_dispatch_ops")
        kimi_k3_ops = load("atom.model_ops.kimi_k3")
        sequence_parallel = load("atom.distributed.sequence_parallel")
        ubatching = load("atom.utils.tbo.ubatching")
        attention_residual = load("atom.model_ops.kimi_k3.attention_residual")
        yield
    kimi = module_dispatch_ops = kimi_k3_ops = sequence_parallel = ubatching = None
    attention_residual = None


WORLD, HIDDEN, LATENT, EXPERTS = 8, 16, 8, 4


class _Collectives:
    """In-process TP group: one thread per rank, collectives meet at a barrier."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(WORLD, timeout=30)
        self.slots = [None] * WORLD

    def _exchange(self, rank, x):
        self.slots[rank] = x
        self.barrier.wait()
        inputs = list(self.slots)
        self.barrier.wait()
        return inputs

    def total(self, rank, x):
        # Round the exact sum once for both all-reduce and reduce-scatter.
        return torch.stack(self._exchange(rank, x)).double().sum(0).to(x.dtype)

    def gather(self, rank, x):
        return torch.cat(self._exchange(rank, x))


@pytest.fixture
def ranks(monkeypatch):
    local = threading.local()

    def reduce_scatter_tensor(x, use_custom=False, dim=0):
        assert dim == 0 and x.shape[0] % WORLD == 0
        return local.comm.total(local.rank, x).chunk(WORLD)[local.rank].clone()

    def all_gather(x, use_custom=False, dim=0):
        assert dim == 0
        return local.comm.gather(local.rank, x)

    group = SimpleNamespace(
        reduce_scatter_tensor=reduce_scatter_tensor,
        all_gather=all_gather,
        device_communicator=SimpleNamespace(
            all_gather=lambda x, dim: all_gather(x, dim=dim),
            ca_comm=SimpleNamespace(
                disabled=False, should_custom_rs=lambda x, dim: True
            ),
            pynccl_comm=SimpleNamespace(
                disabled=False,
                reduce_scatter=lambda out, x: out.copy_(reduce_scatter_tensor(x)),
                group_start=lambda: None,
                group_end=lambda: None,
            ),
        ),
    )
    monkeypatch.setattr(sequence_parallel, "get_tp_group", lambda: group)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_rank", lambda: local.rank)
    monkeypatch.setattr(
        kimi,
        "tensor_model_parallel_all_reduce",
        lambda x: local.comm.total(local.rank, x),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def run(fn):
        comm = _Collectives()

        def on_rank(rank):
            local.rank, local.comm = rank, comm
            return fn(rank)

        with ThreadPoolExecutor(WORLD) as pool:
            return list(pool.map(on_rank, range(WORLD)))

    return run


class _Stub(nn.Module):
    """Stands in for a sub-module class; the test sets `fn` after `__init__`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args, out=None):
        result = self.fn(*args)
        return result if out is None else out.copy_(result)


@pytest.fixture(autouse=True)
def stub_block_parts(monkeypatch):
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_GROUP_RS", "0")
    empty_like = torch.empty_like

    def poisoned_empty_like(*args, **kwargs):
        out = empty_like(*args, **kwargs)
        return out.fill_(float("nan")) if out.is_floating_point() else out

    monkeypatch.setattr(torch, "empty_like", poisoned_empty_like)
    for name in ("ReplicatedLinear", "FusedMoE", "KimiMLP", "RMSNorm"):
        monkeypatch.setattr(kimi, name, _Stub)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: WORLD)
    config = SimpleNamespace(
        enable_tbo=False,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(kimi, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(module_dispatch_ops, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(
        torch.ops.aiter,
        "maybe_dual_stream_split_forward",
        module_dispatch_ops.maybe_dual_stream_split_forward,
    )
    return config


def _weights():
    gen = torch.Generator().manual_seed(3)
    shapes = {
        "gate": (HIDDEN, EXPERTS),
        "down": (HIDDEN, LATENT),
        "up": (LATENT, HIDDEN),
        "shared": (HIDDEN, HIDDEN),
    }
    return {
        name: torch.randn(*shape, generator=gen).to(torch.bfloat16)
        for name, shape in shapes.items()
    }


def _block(w, rank, up_rows):
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_token=2,
        routed_expert_hidden_size=LATENT,
        moe_intermediate_size=HIDDEN,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_shared_experts=1,
        latent_moe_use_norm=True,
        rms_norm_eps=1e-6,
    )
    block = kimi.KimiSparseMoeBlock(
        config, prefix=f"rank{rank}.mlp", alt_stream=object()
    )
    share = (rank + 1) / (WORLD * (WORLD + 1) / 2)
    block.gate.fn = lambda x: x @ w["gate"]
    block.routed_expert_down_proj.fn = lambda x: x @ w["down"]
    block.experts.fn = lambda x, logits: share * torch.tanh(
        x * logits.sum(-1, keepdim=True)
    )
    block.shared_experts.fn = lambda x: share * torch.tanh(x @ w["shared"])
    # KimiMLP's out= path: the same partial, written into `out`.
    shared = block.shared_experts
    shared.down_proj = SimpleNamespace(supports_out=lambda: True)
    shared.gate_up_proj = lambda x: x
    shared.act_and_down_proj = lambda gate_up, out: out.copy_(shared.fn(gate_up))
    # Normalize in FP32 so CPU BF16 rounding is independent of row count.
    block.routed_expert_norm.fn = lambda x: (
        x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    ).to(x.dtype)

    def up_proj(x):
        up_rows.append(x.shape[0])
        return x @ w["up"]

    block.routed_expert_up_proj.fn = up_proj
    return block


def _base_and_op_target(ranks, tokens, target=None):
    w = _weights()
    gen = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)
    target = target or (lambda block, x: block.single_stream_split_moe_forward(x))

    def on_rank(rank):
        up_rows = []
        block = _block(w, rank, up_rows)
        base = block.split_moe_forward(x)
        return base, target(block, x), up_rows

    return ranks(on_rank)


@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_row_sharded_pair_sums_to_the_base_pair(ranks, monkeypatch, tokens):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    results = _base_and_op_target(ranks, tokens)
    for (routed, shared), (summed, zeros), up_rows in results:
        assert up_rows == [tokens, -(-tokens // WORLD)]
        assert summed.shape == zeros.shape == (tokens, HIDDEN)
        assert not zeros.any()
        assert zeros.data_ptr() != summed.data_ptr()
        expected = (routed.float() + shared.float()).to(summed.dtype)
        torch.testing.assert_close(summed, expected)


def test_forward_with_dual_stream_off_takes_the_row_sharded_tail(ranks, monkeypatch):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    tokens = 13
    for (routed, shared), (summed, zeros), up_rows in _base_and_op_target(
        ranks, tokens, lambda block, x: block.forward(x)
    ):
        assert up_rows == [tokens, -(-tokens // WORLD)]
        assert not zeros.any()
        assert zeros.data_ptr() != summed.data_ptr()
        expected = (routed.float() + shared.float()).to(summed.dtype)
        torch.testing.assert_close(summed, expected)


def test_forward_with_tbo_enabled_keeps_the_base_pair(
    ranks, monkeypatch, stub_block_parts
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    stub_block_parts.enable_tbo = True
    tokens = 13
    for (routed, shared), (routed_out, shared_out), up_rows in _base_and_op_target(
        ranks, tokens, lambda block, x: block.forward(x)
    ):
        assert up_rows == [tokens, tokens]
        assert torch.equal(routed_out, routed) and torch.equal(shared_out, shared)


@pytest.mark.parametrize(
    "floor,sharded",
    [(None, False), ("14", False), ("13", True)],
    ids=["default-floor", "below-floor", "at-floor"],
)
def test_row_floor(ranks, monkeypatch, floor, sharded):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    if floor is None:
        monkeypatch.delenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS")
    else:
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", floor)
    tokens = 13
    for (routed, shared), (first, second), up_rows in _base_and_op_target(
        ranks, tokens, lambda block, x: block.forward(x)
    ):
        assert up_rows == [tokens, -(-tokens // WORLD) if sharded else tokens]
        if not sharded:
            assert torch.equal(first, routed) and torch.equal(second, shared)


@pytest.mark.parametrize(
    "threshold,capturing,tbo_step",
    [
        ("13", False, False),
        ("14", False, False),
        ("4", True, False),
        ("4", False, True),
    ],
    ids=["at-threshold", "below-threshold", "graph-capture", "tbo-step"],
)
def test_base_pair_kept_bit_for_bit(ranks, monkeypatch, threshold, capturing, tbo_step):
    tokens = 13
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", threshold)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(ubatching, "tbo_active", lambda: tbo_step)
    results = _base_and_op_target(ranks, tokens)
    for (routed, shared), (routed_out, shared_out), up_rows in results:
        assert up_rows == [tokens, tokens]
        assert torch.equal(routed_out, routed) and torch.equal(shared_out, shared)


BLOCK = 4  # layer 4 closes an attention-residual block, layer 5 does not
_events = threading.local()


def _logs(name, fn):

    def run(x, *args, **kwargs):
        _events.log.append((name, x.shape[0]))
        return fn(x, *args, **kwargs)

    return run


def _torch_attn_res(
    prefix,
    block_residual,
    score_weight,
    eps,
    add_hidden=None,
    out_norm_weight=None,
    out_eps=1e-6,
    add_hidden2=None,
    quant_dtype=None,
    num_blocks=None,
):
    """Attention residual and optional output RMS norm in FP32 torch."""
    assert quant_dtype is None
    if num_blocks is not None:
        block_residual = block_residual[:, :num_blocks]
    ps = prefix.float()
    for add in (add_hidden, add_hidden2):
        if add is not None:
            ps = ps + add.float()
    cands = torch.cat([block_residual.float(), ps[:, None]], 1)
    scores = (cands * score_weight).sum(-1) * torch.rsqrt(cands.pow(2).mean(-1) + eps)
    y = (scores.softmax(-1)[..., None] * cands).sum(1)
    if out_norm_weight is not None:
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + out_eps)
        y = y * out_norm_weight.float()
    return y.to(prefix.dtype), prefix if add_hidden is None else ps.to(prefix.dtype)


class _Attention(nn.Module):
    """This rank's share of a KDA mixer's output, all-reduced unless the layer
    turns `o_proj.reduce_results` off, as RowParallelLinear does."""

    fuse_input_norm_quant = False
    input_quant_prefix = "self_attn.in_proj"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.o_proj = SimpleNamespace(reduce_results=True)

    def forward(self, x):
        _events.log.append(("attention", x.shape[0]))
        out = self.share * torch.tanh(x @ self.w)
        if self.o_proj.reduce_results:
            out = kimi.tensor_model_parallel_all_reduce(out)
        return out


@pytest.fixture
def logged(ranks, monkeypatch):
    group = sequence_parallel.get_tp_group()

    def reduce_scatter(x, use_custom=False, dim=0):
        name = "custom_reduce_scatter" if use_custom else "reduce_scatter"
        _events.log.append((name, x.shape[0]))
        return group.reduce_scatter_tensor(x, use_custom=use_custom, dim=dim)

    comm = group.device_communicator

    def nccl_reduce_scatter(out, x):
        _events.log.append(("pynccl_reduce_scatter", x.shape[0]))
        comm.pynccl_comm.reduce_scatter(out, x)

    def nccl_event(name):
        return lambda: _events.log.append((name, 0))

    logged_group = SimpleNamespace(
        reduce_scatter_tensor=reduce_scatter,
        all_gather=_logs("all_gather", group.all_gather),
        device_communicator=SimpleNamespace(
            all_gather=_logs("custom_all_gather", comm.all_gather),
            ca_comm=comm.ca_comm,
            pynccl_comm=SimpleNamespace(
                disabled=False,
                reduce_scatter=nccl_reduce_scatter,
                group_start=nccl_event("nccl_group_start"),
                group_end=nccl_event("nccl_group_end"),
            ),
        ),
    )
    monkeypatch.setattr(sequence_parallel, "get_tp_group", lambda: logged_group)
    all_reduce = _logs("all_reduce", kimi.tensor_model_parallel_all_reduce)
    monkeypatch.setattr(kimi, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        kimi_k3_ops, "apply_attn_res", _logs("attn_res", _torch_attn_res)
    )
    monkeypatch.setattr(
        torch.ops.aiter,
        "kimi_k3_post_attention_forward",
        module_dispatch_ops.kimi_k3_post_attention_forward,
    )

    def run(fn):
        def on_rank(rank):
            _events.log = []
            return fn(rank), _events.log

        return ranks(on_rank)

    return run


def _layers(layer_num, attention=None, kda=True):
    w = _weights()
    gen = torch.Generator().manual_seed(layer_num)
    w_attn = (torch.randn(HIDDEN, HIDDEN, generator=gen) / HIDDEN**0.5).bfloat16()
    score_weights = torch.randn(2, HIDDEN, generator=gen)
    norm_weights = (1 + 0.1 * torch.randn(2, HIDDEN, generator=gen)).bfloat16()
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        rms_norm_eps=1e-6,
        num_experts=4,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        kimi_kda_layers=[layer_num] if kda else [],
        attn_res_block_size=BLOCK,
    )
    layers = []
    for rank in range(WORLD):
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
            block = _block(w, rank, [])
            for name in (
                "gate",
                "routed_expert_down_proj",
                "experts",
                "routed_expert_norm",
                "routed_expert_up_proj",
                "shared_experts",
            ):
                stub = getattr(block, name)
                stub.fn = _logs(name, stub.fn)
            patch.setattr(kimi, "KimiSparseMoeBlock", lambda *a, b=block, **k: b)
            patch.setattr(kimi, "KimiKDAAttention", attention or _Attention)
            patch.setattr(kimi, "KimiFullAttention", attention or _Attention)
            layer = kimi.KimiDecoderLayer(
                SimpleNamespace(hf_config=config, quant_config=None),
                prefix=f"rank{rank}.layers.{layer_num}",
                layer_num=layer_num,
            )
        layer.self_attn.w = w_attn
        layer.self_attn.share = (rank + 1) / (WORLD * (WORLD + 1) / 2)
        layer.self_attention_attn_res.score_weight = score_weights[0]
        layer.mlp_attn_res.score_weight = score_weights[1]
        layer.input_layernorm.weight = norm_weights[0]
        layer.post_attention_layernorm.weight = norm_weights[1]
        layers.append(layer)
    return layers


def _inputs(tokens):
    gen = torch.Generator().manual_seed(tokens)
    x, add, add2 = torch.randn(3, tokens, HIDDEN, generator=gen).bfloat16()
    bank = torch.randn(tokens, 2, HIDDEN, generator=gen).bfloat16()
    return x, bank, add, add2


def _base_forward(layer, x, bank, add, add2):
    """Replicated attention reduction, residual and FFN before the new op."""
    o_proj = layer.self_attn.o_proj
    reduce_results, o_proj.reduce_results = o_proj.reduce_results, True
    try:
        h, prefix = layer.self_attention_attn_res(x, bank, add, add2)
        bank, prefix = layer.self_attention_attn_res.maybe_close_block(prefix, bank)
        h, prefix = layer.mlp_attn_res(prefix, bank, layer.self_attn(h))
        h, shared = layer._ffn(h)
    finally:
        o_proj.reduce_results = reduce_results
    return prefix, h, shared, bank


@pytest.mark.parametrize("layer_num", [4, 5], ids=["block-closes", "mid-block"])
@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_row_sharded_layer_sums_to_the_base(logged, monkeypatch, tokens, layer_num):
    layers = _layers(layer_num)
    inputs = _inputs(tokens)
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "100000")
    base = logged(
        lambda rank: _base_forward(layers[rank], *(x.clone() for x in inputs))
    )
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    rows = logged(lambda rank: layers[rank](None, *(x.clone() for x in inputs)))

    chunk = -(-tokens // WORLD)
    padded = chunk * WORLD
    for (want, _), (got, log) in zip(base, rows):
        assert log == [
            ("attn_res", tokens),
            ("attention", tokens),
            ("reduce_scatter", padded),
            ("attn_res", chunk),
            ("all_gather", chunk),
            ("gate", tokens),
            ("routed_expert_down_proj", tokens),
            ("experts", tokens),
            ("reduce_scatter", padded),
            ("routed_expert_norm", chunk),
            ("routed_expert_up_proj", chunk),
            ("shared_experts", tokens),
            ("reduce_scatter", padded),
            ("all_gather", chunk),
        ]
        residual, routed, shared, bank = got
        assert not routed.any() and not shared.any()
        assert len({t.data_ptr() for t in (residual, routed, shared)}) == 3
        assert torch.equal(bank, want[3])
        # The row-sharded sum rounds to BF16 twice, the reference once.
        exact = sum(t.float() for t in want[:3])
        bound = 3 * 2**-8 * sum(t.float().abs() for t in want[:3])
        assert ((residual.float() - exact).abs() <= bound).all()
        assert torch.equal(kimi.KimiDecoderLayer.aux_hidden_state(got), residual)
        cloned = tuple(x.clone() for x in got)
        assert torch.equal(kimi.KimiDecoderLayer.aux_hidden_state(cloned), residual)


@pytest.mark.parametrize(
    "threshold,capturing,tbo_step",
    [("13", False, False), ("4", True, False), ("4", False, True)],
    ids=["at-threshold", "graph-capture", "tbo-step"],
)
@pytest.mark.parametrize("layer_num", [4, 5], ids=["block-closes", "mid-block"])
def test_base_kernels_launch_in_order(
    logged, monkeypatch, threshold, capturing, tbo_step, layer_num
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", threshold)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(ubatching, "tbo_active", lambda: tbo_step)
    layers = _layers(layer_num)
    inputs = _inputs(13)
    base = logged(
        lambda rank: _base_forward(layers[rank], *(x.clone() for x in inputs))
    )
    got = logged(lambda rank: layers[rank](None, *(x.clone() for x in inputs)))
    for (want, want_log), (out, log) in zip(base, got):
        assert log == want_log
        assert log.count(("all_reduce", 13)) == 3
        assert all(torch.equal(a, b) for a, b in zip(out, want))


@pytest.mark.parametrize("latent", ["0", "1"], ids=["latent-off", "latent-on"])
@pytest.mark.parametrize("floor", [None, "14"], ids=["default-floor", "below-floor"])
def test_below_row_floor_all_reduces(logged, monkeypatch, floor, latent):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", latent)
    if floor is None:
        monkeypatch.delenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS")
    else:
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", floor)
    layers = _layers(BLOCK + 1)
    inputs = _inputs(13)
    base = logged(
        lambda rank: _base_forward(layers[rank], *(x.clone() for x in inputs))
    )
    got = logged(lambda rank: layers[rank](None, *(x.clone() for x in inputs)))
    for (want, want_log), (out, log) in zip(base, got):
        assert log == want_log
        assert all(torch.equal(a, b) for a, b in zip(out, want))


def _merge_front(monkeypatch, blocks):
    """Give each block the merged [gate_up | gate | down] front weight of its
    stubs, and a logged GEMM over it."""
    w = _weights()
    gen = torch.Generator().manual_seed(5)
    gate_up = torch.randn(3, HIDDEN, generator=gen).bfloat16()
    merged = torch.cat([gate_up, w["gate"].t(), w["down"].t()])
    for block in blocks:
        block.front_proj.merged_weight = merged
    gemm = _logs("front_rows", lambda x, weight, otype: x @ weight.t())
    monkeypatch.setattr(kimi, "tgemm", SimpleNamespace(mm=gemm))


def test_latent_row_shard_is_on_unless_0(monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", raising=False)
    assert kimi.envs.ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", "0")
    assert not kimi.envs.ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD


def test_row_tail_defaults(monkeypatch):
    """Eager steps from 2048 rows and token-graph buckets from 512 take the row
    tail, on the custom collectives unless ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM=0."""
    for name in (
        "ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS",
        "ATOM_KIMI_K3_MOE_ROW_SHARD_GRAPH_MIN_TOKENS",
        "ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM",
    ):
        monkeypatch.delenv(name, raising=False)
    assert kimi.envs.ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS == 2048
    assert kimi.envs.ATOM_KIMI_K3_MOE_ROW_SHARD_GRAPH_MIN_TOKENS == 512
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM is True
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM", "0")
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM is False


@pytest.mark.parametrize("layer_num", [4, 5], ids=["block-closes", "mid-block"])
@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_latent_row_shard_gathers_router_and_latent_rows(
    logged, monkeypatch, tokens, layer_num
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    layers = _layers(layer_num)
    _merge_front(monkeypatch, [layer.block_sparse_moe for layer in layers])
    inputs = _inputs(tokens)

    def forward(rank):
        return layers[rank](None, *(x.clone() for x in inputs))

    replicated = logged(forward)
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", "1")
    sharded = logged(forward)

    chunk = -(-tokens // WORLD)
    for (want, want_log), (got, log) in zip(replicated, sharded):
        front = want_log.index(("gate", tokens))
        assert want_log[front + 1] == ("routed_expert_down_proj", tokens)
        assert log == (
            want_log[:front]
            + [("front_rows", chunk), ("all_gather", chunk)]
            + want_log[front + 2 :]
        )
        torch.testing.assert_close(got[0], want[0])
        assert not got[1].any() and not got[2].any()


@pytest.mark.parametrize(
    "latent,merged", [("0", True), ("1", False)], ids=["off", "unmerged"]
)
def test_latent_row_shard_off_or_unmerged_keeps_the_row_tail_bit_for_bit(
    logged, monkeypatch, latent, merged
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    layers = _layers(BLOCK)
    inputs = _inputs(13)

    def forward(rank):
        return layers[rank](None, *(x.clone() for x in inputs))

    base = logged(forward)
    if merged:
        _merge_front(monkeypatch, [layer.block_sparse_moe for layer in layers])
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", latent)
    for (want, want_log), (out, log) in zip(base, logged(forward)):
        assert log == want_log
        assert all(torch.equal(a, b) for a, b in zip(out, want))


@pytest.mark.parametrize("latent", ["0", "1"], ids=["latent-off", "latent-on"])
@pytest.mark.parametrize("skip", ["0", "1"], ids=["zero-row-off", "zero-row-on"])
def test_row_tail_returns_the_registered_zero_pair(logged, monkeypatch, latent, skip):
    """The next residual mix reads a zero row only for the registered pair
    itself, so the row tail must return that pair whichever way the MoE
    front runs."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    monkeypatch.setenv("ATOM_KIMI_K3_ATTN_RES_SKIP_ZERO_ADDENDS", skip)
    layers = _layers(BLOCK)
    _merge_front(monkeypatch, [layer.block_sparse_moe for layer in layers])
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", latent)
    assert kimi.zero_addends is attention_residual.zero_addends
    lock, local = threading.Lock(), threading.local()

    def zero_addends(like):
        # Ranks are threads here and share the one registration.
        with lock:
            local.pair = attention_residual.zero_addends(like)
            local.held = list(attention_residual._ZERO_ADDENDS)
        return local.pair

    monkeypatch.setattr(kimi, "zero_addends", zero_addends)
    inputs = _inputs(13)

    def forward(rank):
        out = layers[rank](None, *(x.clone() for x in inputs))
        return out, local.pair, local.held

    try:
        for (out, pair, held), log in logged(forward):
            assert ("all_gather", 2) in log
            assert (("front_rows", 2) in log) == (latent == "1")
            assert out[1] is pair[0] and out[2] is pair[1]
            assert not out[1].any() and not out[2].any()
            if skip == "1":
                assert len(held) == 2 and held[0] is pair[0] and held[1] is pair[1]
            else:
                assert not held
    finally:
        attention_residual._ZERO_ADDENDS.clear()


@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_latent_row_shard_in_the_split_op_shards_its_input(logged, monkeypatch, tokens):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    w = _weights()
    blocks = [_block(w, rank, []) for rank in range(WORLD)]
    _merge_front(monkeypatch, blocks)
    gen = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)

    def forward(rank):
        return blocks[rank].single_stream_split_moe_forward(x)

    replicated = logged(forward)
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", "1")
    sharded = logged(forward)

    chunk = -(-tokens // WORLD)
    for (want, want_log), (got, log) in zip(replicated, sharded):
        assert log == [("front_rows", chunk), ("all_gather", chunk)] + want_log
        torch.testing.assert_close(got[0], want[0])
        assert not got[1].any()


@pytest.mark.parametrize(
    "tokens,threshold,floor,expected",
    [
        (2047, None, None, False),
        (2048, None, None, True),
        (2048, "2048", None, False),
        (2049, "2048", None, True),
    ],
)
def test_row_gate_intersects_floor_and_strict_stream_threshold(
    ranks, monkeypatch, tokens, threshold, floor, expected
):
    for name, value in [
        ("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", threshold),
        ("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", floor),
    ]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert (
        kimi.KimiSparseMoeBlock.takes_row_tail(SimpleNamespace(tp_size=WORLD), tokens)
        is expected
    )


def test_row_gate_refuses_graph_warmup(ranks, monkeypatch):
    monkeypatch.delenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", raising=False)
    monkeypatch.delenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", raising=False)
    block = SimpleNamespace(tp_size=WORLD)
    assert kimi.KimiSparseMoeBlock.takes_row_tail(block, 8192)
    monkeypatch.setattr(
        kimi, "get_forward_context", lambda: SimpleNamespace(in_hipgraph=True)
    )
    assert not kimi.KimiSparseMoeBlock.takes_row_tail(block, 8192)


def _in_graph(monkeypatch, token_graph=True):
    """This forward records a prefill token-graph bucket or warms one up, or
    with ``token_graph`` False is inside some other graph capture."""
    monkeypatch.setattr(
        kimi,
        "get_forward_context",
        lambda: SimpleNamespace(in_hipgraph=True, context=None, tr_graph="record"),
    )
    monkeypatch.setattr(kimi, "prefill_token_graph_capture", lambda: token_graph)


@pytest.mark.parametrize(
    "graph_floor,token_graph,tokens,expected",
    [
        (None, True, 511, False),
        (None, True, 512, True),
        ("0", True, 4096, False),
        ("1024", True, 1023, False),
        ("1024", True, 1024, True),
        ("1024", False, 4096, False),
    ],
    ids=[
        "below-default",
        "at-default",
        "off",
        "below-floor",
        "at-floor",
        "other-capture",
    ],
)
def test_row_gate_in_a_graph_takes_the_graph_floor(
    ranks, monkeypatch, graph_floor, token_graph, tokens, expected
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", "0")
    if graph_floor is None:
        monkeypatch.delenv("ATOM_KIMI_K3_MOE_ROW_SHARD_GRAPH_MIN_TOKENS", raising=False)
    else:
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_GRAPH_MIN_TOKENS", graph_floor)
    _in_graph(monkeypatch, token_graph)
    block = SimpleNamespace(tp_size=WORLD)
    assert kimi.KimiSparseMoeBlock.takes_row_tail(block, tokens) is expected


def test_graph_floor_leaves_eager_steps_alone(ranks, monkeypatch):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", "8192")
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_GRAPH_MIN_TOKENS", "1024")
    monkeypatch.setattr(kimi, "prefill_token_graph_capture", lambda: True)
    block = SimpleNamespace(tp_size=WORLD)
    assert not kimi.KimiSparseMoeBlock.takes_row_tail(block, 4096)
    assert kimi.KimiSparseMoeBlock.takes_row_tail(block, 8192)


@pytest.mark.parametrize("latent", ["0", "1"], ids=["latent-off", "latent-on"])
@pytest.mark.parametrize("how", ["switch", "token-graph"])
@pytest.mark.parametrize("tokens", [13, 16])
def test_custom_collectives_run_the_same_row_path(
    logged, monkeypatch, how, latent, tokens
):
    """Under the switch, and always in a token-graph recording, every
    reduce-scatter and gather of the row path takes the custom kernels. The
    stub collectives sum alike, so the outputs match bit for bit."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", latent)
    layers = _layers(BLOCK + 1)
    _merge_front(monkeypatch, [layer.block_sparse_moe for layer in layers])
    inputs = _inputs(tokens)

    def forward(rank):
        return layers[rank](None, *(x.clone() for x in inputs))

    rccl = logged(forward)
    if how == "switch":
        monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM", "1")
    else:
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", "100000")
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_GRAPH_MIN_TOKENS", "8")
        _in_graph(monkeypatch)
    custom = logged(forward)
    renamed = {
        "reduce_scatter": "custom_reduce_scatter",
        "all_gather": "custom_all_gather",
    }
    for (want, want_log), (got, log) in zip(rccl, custom):
        assert ("reduce_scatter", -(-tokens // WORLD) * WORLD) in want_log
        assert log == [(renamed.get(name, name), rows) for name, rows in want_log]
        assert all(torch.equal(a, b) for a, b in zip(got, want))


# 314 bytes: above a 16-row partial of HIDDEN (512 B), below one of LATENT (256 B)
_PYNCCL_RS_MIB = "0.0003"


def _comm_log(log):
    return [
        event
        for event in log
        if event[0].endswith("reduce_scatter") or event[0].startswith("nccl_group")
    ]


def _row_layer_forward(monkeypatch, latent, tokens):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", latent)
    layers = _layers(BLOCK + 1)
    _merge_front(monkeypatch, [layer.block_sparse_moe for layer in layers])
    inputs = _inputs(tokens)

    def forward(rank):
        return layers[rank](None, *(x.clone() for x in inputs))

    return forward


def test_pynccl_cap_and_group_defaults(monkeypatch):
    """Row-tail reduce-scatters above 48 MiB run on pynccl and eager steps group
    the routed and shared ones, unless set to 0."""
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", raising=False)
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_SHARD_GROUP_RS", raising=False)
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB == 48
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_GROUP_RS is True
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_GROUP_RS", "0")
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB == 0
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_GROUP_RS is False


@pytest.mark.parametrize("latent", ["0", "1"], ids=["latent-off", "latent-on"])
@pytest.mark.parametrize("tokens", [13, 16])
def test_pynccl_cap_moves_larger_row_scatters_to_pynccl(
    logged, monkeypatch, latent, tokens
):
    """Above the cap the attention and shared reduce-scatters run on pynccl,
    the smaller latent one stays on the custom kernel; same sums."""
    forward = _row_layer_forward(monkeypatch, latent, tokens)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM", "1")
    custom = logged(forward)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", _PYNCCL_RS_MIB)
    capped = logged(forward)
    rows = -(-tokens // WORLD) * WORLD
    for (want, want_log), (got, log) in zip(custom, capped):
        assert _comm_log(want_log) == [("custom_reduce_scatter", rows)] * 3
        assert _comm_log(log) == [
            ("pynccl_reduce_scatter", rows),
            ("custom_reduce_scatter", rows),
            ("pynccl_reduce_scatter", rows),
        ]
        assert all(torch.equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize("latent", ["0", "1"], ids=["latent-off", "latent-on"])
@pytest.mark.parametrize("tokens", [13, 16])
def test_group_rs_scatters_routed_and_shared_as_one_nccl_group(
    logged, monkeypatch, latent, tokens
):
    """With the shared reduce-scatter on pynccl, the routed one joins it in
    one NCCL group after the shared experts ran; same sums. Without (the
    shared one on the custom kernel) both stay apart."""
    forward = _row_layer_forward(monkeypatch, latent, tokens)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM", "1")
    base = logged(forward)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_GROUP_RS", "1")
    apart = logged(forward)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", _PYNCCL_RS_MIB)
    grouped = logged(forward)
    rows = -(-tokens // WORLD) * WORLD
    for (want, _), (got_apart, log_apart), (got, log) in zip(base, apart, grouped):
        assert _comm_log(log_apart) == [("custom_reduce_scatter", rows)] * 3
        assert _comm_log(log) == [
            ("pynccl_reduce_scatter", rows),
            ("nccl_group_start", 0),
            ("pynccl_reduce_scatter", rows),
            ("pynccl_reduce_scatter", rows),
            ("nccl_group_end", 0),
        ]
        names = [name for name, _ in log]
        assert names.index("shared_experts") < names.index("nccl_group_start")
        assert all(torch.equal(a, b) for a, b in zip(got_apart, want))
        assert all(torch.equal(a, b) for a, b in zip(got, want))


def test_group_rs_stays_apart_in_token_graphs(logged, monkeypatch):
    """A token-graph recording keeps the cap but never groups."""
    forward = _row_layer_forward(monkeypatch, "1", 16)
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", "100000")
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_GRAPH_MIN_TOKENS", "8")
    _in_graph(monkeypatch)
    base = logged(forward)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", _PYNCCL_RS_MIB)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_GROUP_RS", "1")
    graph = logged(forward)
    for (want, _), (got, log) in zip(base, graph):
        assert _comm_log(log) == [
            ("pynccl_reduce_scatter", 16),
            ("custom_reduce_scatter", 16),
            ("pynccl_reduce_scatter", 16),
        ]
        assert all(torch.equal(a, b) for a, b in zip(got, want))


def test_pynccl_cap_and_group_need_the_custom_collectives(logged, monkeypatch):
    """On RCCL (ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM=0) both switches do nothing."""
    forward = _row_layer_forward(monkeypatch, "1", 16)
    base = logged(forward)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", _PYNCCL_RS_MIB)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_GROUP_RS", "1")
    rccl = logged(forward)
    for (want, want_log), (got, log) in zip(base, rccl):
        assert log == want_log
        assert _comm_log(log) == [("reduce_scatter", 16)] * 3
        assert all(torch.equal(a, b) for a, b in zip(got, want))


@pytest.fixture
def side(monkeypatch):
    """`_side_comm` on the logged pynccl stub, its side stream run inline."""

    def run(fn, *uses):
        _events.log.append(("side_stream", 0))
        fn()
        return lambda: _events.log.append(("side_wait", 0))

    def side_comm():
        nccl = sequence_parallel.get_tp_group().device_communicator.pynccl_comm
        return nccl, SimpleNamespace(run=run)

    monkeypatch.setattr(sequence_parallel, "_side_comm", side_comm)


def test_side_rs_defaults(monkeypatch):
    """The side reduce-scatter is off unless 1, from 4097 rows."""
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_SHARD_SIDE_RS", raising=False)
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_SHARD_SIDE_RS_MIN_TOKENS", raising=False)
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_SIDE_RS is False
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_SIDE_RS_MIN_TOKENS == 4097
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_SIDE_RS", "1")
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_SIDE_RS is True


@pytest.mark.parametrize("latent", ["0", "1"], ids=["latent-off", "latent-on"])
@pytest.mark.parametrize("tokens", [13, 16])
def test_side_rs_reduces_shared_beside_the_routed_experts(
    logged, side, monkeypatch, latent, tokens
):
    """The shared experts run first and their reduce-scatter is queued on the
    side stream before the routed experts, in place of the NCCL group; the
    main stream waits for it after the routed scatter. Same sums as the two
    scatters apart on the main stream."""
    forward = _row_layer_forward(monkeypatch, latent, tokens)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM", "1")
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", _PYNCCL_RS_MIB)
    apart = logged(forward)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_GROUP_RS", "1")
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_SIDE_RS", "1")
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_SIDE_RS_MIN_TOKENS", str(tokens))
    sided = logged(forward)
    rows = -(-tokens // WORLD) * WORLD
    for (want, _), (got, log) in zip(apart, sided):
        names = [name for name, _ in log]
        queued = names.index("side_stream")
        assert names.index("shared_experts") < queued < names.index("experts")
        assert log[queued + 1] == ("pynccl_reduce_scatter", rows)
        assert names.index("routed_expert_up_proj") < names.index("side_wait")
        assert "nccl_group_start" not in names
        assert _comm_log(log) == [
            ("pynccl_reduce_scatter", rows),
            ("pynccl_reduce_scatter", rows),
            ("custom_reduce_scatter", rows),
        ]
        assert all(torch.equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize("case", ["below-floor", "rccl", "token-graph"])
def test_side_rs_gate(logged, side, monkeypatch, case):
    """Below its row floor, on RCCL or in a token-graph recording the switch
    leaves the row tail as it was."""
    forward = _row_layer_forward(monkeypatch, "1", 16)
    if case != "rccl":
        monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_CUSTOM_COMM", "1")
    if case == "token-graph":
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", "100000")
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_GRAPH_MIN_TOKENS", "8")
        _in_graph(monkeypatch)
    base = logged(forward)
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_SIDE_RS", "1")
    floor = "17" if case == "below-floor" else "16"
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_SIDE_RS_MIN_TOKENS", floor)
    gated = logged(forward)
    for (want, want_log), (got, log) in zip(base, gated):
        assert log == want_log
        assert all(torch.equal(a, b) for a, b in zip(got, want))


def test_side_stream_runs_after_the_queued_work(monkeypatch):
    """The side stream (priority -1) waits for the current stream, runs the
    work and records its end there, keeps the inputs alive for it, and the
    wait lands on the stream current when it is called."""
    log = []

    class Stream:
        def __init__(self, name, priority=0):
            self.name, self.priority = name, priority

        def wait_stream(self, other):
            log.append((self.name, "wait_stream", other.name))

        def wait_event(self, event):
            log.append((self.name, "wait_event", event.stream))

    current = [Stream("main")]

    class Event:
        def record(self):
            self.stream = current[-1].name
            log.append((self.stream, "record"))

    @contextmanager
    def on(stream):
        current.append(stream)
        yield
        current.pop()

    monkeypatch.setattr(
        torch.cuda, "Stream", lambda priority=0: Stream("side", priority)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current[-1])
    monkeypatch.setattr(torch.cuda, "stream", on)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    side = sequence_parallel._SideStream()
    assert side.stream.priority == -1

    class Held:
        def record_stream(self, s):
            log.append(("x", "record_stream", s.name))

    x = Held()
    alive = weakref.ref(x)
    wait = side.run(lambda: log.append((current[-1].name, "work")), x)
    del x
    gc.collect()
    assert alive() is not None, "the input must outlive the side work"
    current[0] = Stream("later")
    wait()
    gc.collect()
    assert alive() is None, "the wait lets go of the input"
    assert log == [
        ("side", "wait_stream", "main"),
        ("side", "work"),
        ("side", "record"),
        ("later", "wait_event", "side"),
    ]


@pytest.mark.parametrize("own", [False, True], ids=["tp-comm", "own-comm"])
def test_side_comm_is_the_tp_groups_unless_asked(monkeypatch, own):
    """The side reduce runs on the TP group's pynccl communicator (no second
    set of RCCL buffers); ..._SIDE_RS_OWN_COMM=1 builds one of its own, once."""
    tp_comm = SimpleNamespace(disabled=False)
    tp = SimpleNamespace(
        cpu_group="cpu",
        device="dev",
        device_communicator=SimpleNamespace(pynccl_comm=tp_comm),
    )
    built = []

    def own_comm(group, device):
        built.append((group, device))
        return SimpleNamespace(disabled=False)

    monkeypatch.setattr(sequence_parallel, "get_tp_group", lambda: tp)
    name = "aiter.dist.device_communicators.communicator_pynccl"
    for parent in ("aiter", "aiter.dist", "aiter.dist.device_communicators"):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, SimpleNamespace(__path__=[]))
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(PyNcclCommunicator=own_comm))
    monkeypatch.setattr(sequence_parallel, "_SideStream", lambda: "side")
    monkeypatch.setattr(sequence_parallel, "_SIDE", [])
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_SIDE_RS_OWN_COMM", "1" if own else "0")
    nccl, stream = sequence_parallel._side_comm()
    assert sequence_parallel._side_comm() == (nccl, stream)
    assert stream == "side"
    if own:
        assert nccl is not tp_comm and built == [("cpu", "dev")]
    else:
        assert nccl is tp_comm and built == []


def test_side_comm_refuses_without_a_tp_pynccl(monkeypatch):
    tp = SimpleNamespace(device_communicator=SimpleNamespace(pynccl_comm=None))
    monkeypatch.setattr(sequence_parallel, "get_tp_group", lambda: tp)
    monkeypatch.setattr(sequence_parallel, "_SIDE", [])
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_SHARD_SIDE_RS_OWN_COMM", raising=False)
    with pytest.raises(RuntimeError, match="pynccl"):
        sequence_parallel._side_comm()


@pytest.mark.parametrize("latent", ["0", "1"], ids=["latent-off", "latent-on"])
@pytest.mark.parametrize("layer_num", [4, 5], ids=["block-closes", "mid-block"])
def test_token_graph_padding_rows_never_reach_real_rows(
    logged, monkeypatch, latent, layer_num
):
    """A step padded to its bucket replays the row path over every bucket row:
    whatever the padding rows hold, the real rows come out the same."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_LATENT_ROW_SHARD", latent)
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_GRAPH_MIN_TOKENS", "16")
    _in_graph(monkeypatch)
    layers = _layers(layer_num)
    _merge_front(monkeypatch, [layer.block_sparse_moe for layer in layers])
    real, bucket = 13, 16
    padded = _inputs(bucket)
    other = tuple(x.clone() for x in padded)
    gen = torch.Generator().manual_seed(7)
    for x in other:
        x[real:] = 4 * torch.randn(x[real:].shape, generator=gen).to(x.dtype)
    runs = [
        logged(
            lambda rank, inputs=inputs: layers[rank](None, *(x.clone() for x in inputs))
        )
        for inputs in (padded, other)
    ]
    for (a, log), (b, _) in zip(*runs):
        assert ("custom_reduce_scatter", bucket) in log
        assert not torch.equal(a[0][real:], b[0][real:])
        assert all(torch.equal(x[:real], y[:real]) for x, y in zip(a, b))


@pytest.mark.parametrize(
    "world,latent,shared,tbo,eligible",
    [
        (WORLD, LATENT, 1, False, True),
        (1, LATENT, 1, False, False),
        (WORLD, None, 1, False, False),
        (WORLD, LATENT, 0, False, False),
        (WORLD, LATENT, 1, True, False),
    ],
)
def test_constructor_limits_row_dispatch(
    monkeypatch, stub_block_parts, world, latent, shared, tbo, eligible
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    config = _block(_weights(), 0, []).config
    config.routed_expert_hidden_size, config.num_shared_experts = latent, shared
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: world)
    stub_block_parts.enable_tbo = tbo
    block = kimi.KimiSparseMoeBlock(config, prefix="scope", alt_stream=object())
    assert block.row_shardable is eligible and block._use_dispatch_op is eligible


def test_mono_fallback_uses_post_attention_reduction():
    path = Path(kimi.__file__).parent / "mono/runner.py"
    fn = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_fallback"
    )
    namespace = {}
    exec(  # noqa: S102 - execute only the checked-in fallback method
        compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace
    )
    prefix, normed, bank = (
        torch.ones(2, 4),
        torch.full((2, 4), 2.0),
        torch.zeros(2, 1, 4),
    )
    site = _Stub()
    site.fn = lambda *args: (normed, prefix)
    site.maybe_close_block = lambda prefix, blocks: (blocks, prefix)
    layer = SimpleNamespace(
        self_attention_attn_res=site,
        is_linear_attn=True,
        self_attn=lambda x: x + 1,
        _post_attention=lambda *args: args,
    )
    layer.forward = MethodType(kimi.KimiDecoderLayer.forward, layer)
    runner = SimpleNamespace(inner=SimpleNamespace(layers=[layer]))
    partial, residual, blocks = namespace["_fallback"](
        runner, SimpleNamespace(layer_id=0), None, prefix, bank, None, None
    )
    assert torch.equal(partial, normed + 1) and residual is prefix
    assert torch.equal(blocks, bank)


@pytest.mark.parametrize("shape", [(13, 6), (13, 3, 4)], ids=["2d", "3d"])
@pytest.mark.parametrize("missing", [0, 3])
def test_pad_in_place_pads_like_the_full_pad(monkeypatch, shape, missing):
    x = torch.randn(shape).bfloat16()
    x[0] = -0.0
    monkeypatch.setenv("ATOM_ROW_SHARD_PAD_IN_PLACE", "0")
    want = sequence_parallel._pad_rows(x, missing)
    monkeypatch.setenv("ATOM_ROW_SHARD_PAD_IN_PLACE", "1")
    got = sequence_parallel._pad_rows(x, missing)
    assert got.shape == want.shape and got.stride() == want.stride()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("on", ["0", "1"], ids=["off", "on"])
@pytest.mark.parametrize("tokens", [13, 16])
def test_padded_rows_reach_the_reduce_scatter_uncopied(monkeypatch, on, tokens):
    seen = []

    def reduce_scatter_tensor(x, use_custom=False, dim=0):
        seen.append(x)
        return x.chunk(WORLD)[0].clone()

    group = SimpleNamespace(reduce_scatter_tensor=reduce_scatter_tensor)
    monkeypatch.setattr(sequence_parallel, "get_tp_group", lambda: group)
    monkeypatch.setenv("ATOM_ROW_SHARD_PAD_IN_PLACE", on)
    rows = sequence_parallel.SequenceParallel(tokens, WORLD, 0)
    out = rows.padded_rows(torch.ones(tokens, 5, dtype=torch.bfloat16))
    if on == "0" or tokens % WORLD == 0:
        assert out is None
        return
    assert out.shape == (tokens, 5) and out.is_contiguous()
    out.copy_(torch.randn(tokens, 5))
    rows.scatter(out)
    (x,) = seen
    assert x.shape == (16, 5) and x.data_ptr() == out.data_ptr()
    assert torch.equal(x[:tokens], out) and not x[tokens:].any()


def _bits(tensors):
    return [t.view(torch.int16) for t in tensors]


@pytest.mark.parametrize("layer_num", [4, 5], ids=["block-closes", "mid-block"])
@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_pad_in_place_keeps_the_row_tail_bit_for_bit(
    logged, monkeypatch, tokens, layer_num
):
    """With pad rows, the routed and shared partials are written into padded
    buffers that their reduce-scatters take as they are; the o_proj partial
    is copied into one. Outputs and kernels stay the same."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    layers = _layers(layer_num)
    inputs = _inputs(tokens)
    monkeypatch.setenv("ATOM_ROW_SHARD_PAD_IN_PLACE", "0")
    base = logged(lambda rank: layers[rank](None, *(x.clone() for x in inputs)))

    pad_rows, local = sequence_parallel._pad_rows, threading.local()

    def recorded(x, missing):
        out = pad_rows(x, missing)
        if x.shape[0] == tokens:  # a reduce-scatter's input, not a shard
            local.pads.append((x.shape[-1], out is getattr(x, "_padded_rows_of", 0)))
        return out

    monkeypatch.setattr(sequence_parallel, "_pad_rows", recorded)
    monkeypatch.setenv("ATOM_ROW_SHARD_PAD_IN_PLACE", "1")

    def forward(rank):
        local.pads = []
        return layers[rank](None, *(x.clone() for x in inputs)), local.pads

    padded = tokens % WORLD != 0
    for (want, want_log), ((got, pads), log) in zip(base, logged(forward)):
        assert log == want_log
        assert all(map(torch.equal, _bits(got), _bits(want)))
        assert pads == [(HIDDEN, False), (LATENT, padded), (HIDDEN, padded)]


class _Stream:
    """CPU stand-in for a CUDA stream: logs its joins."""

    def __init__(self, name: str) -> None:
        self.name = name

    def wait_stream(self, other) -> None:
        _events.log.append((f"{self.name} waits {other.name}", 0))


class _OnStream:
    def __init__(self, stream) -> None:
        self.stream = stream

    def __enter__(self):
        _events.log.append((f"on {self.stream.name}", 0))

    def __exit__(self, *exc):
        _events.log.append((f"off {self.stream.name}", 0))


@pytest.mark.parametrize("layer_num", [4, 5], ids=["block-closes", "mid-block"])
@pytest.mark.parametrize("tokens", [8, 13])
def test_zeros_beside_gather_are_written_on_alt_while_it_runs(
    logged, monkeypatch, tokens, layer_num
):
    """The zero addends are allocated and filled on the second stream after
    the tail's last add and before its gather, and the layer waits for them
    before it returns: same outputs, the gather in between."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    monkeypatch.setenv("ATOM_KIMI_K3_ATTN_RES_SKIP_ZERO_ADDENDS", "0")
    layers = _layers(layer_num)
    inputs = _inputs(tokens)
    base = logged(lambda rank: layers[rank](None, *(x.clone() for x in inputs)))

    main, alt = _Stream("main"), _Stream("alt")
    monkeypatch.setattr(
        kimi.KimiDecoderLayer, "_zero_addends_stream", lambda self, like: alt
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", _OnStream)
    chunk = -(-tokens // WORLD)
    rows = logged(lambda rank: layers[rank](None, *(x.clone() for x in inputs)))
    for (want, want_log), (got, log) in zip(base, rows):
        assert all(map(torch.equal, _bits(got), _bits(want)))
        routed, shared = got[1:3]
        assert not routed.any() and not shared.any()
        assert routed.data_ptr() != shared.data_ptr()
        assert want_log[-1] == ("all_gather", chunk)
        assert log == want_log[:-1] + [
            ("alt waits main", 0),
            ("on alt", 0),
            ("off alt", 0),
            ("all_gather", chunk),
            ("main waits alt", 0),
        ]


@pytest.mark.parametrize(
    "on,alt,cuda,graph,capturing,stream",
    [
        ("1", True, True, False, False, True),
        ("0", True, True, False, False, False),
        ("1", False, True, False, False, False),
        ("1", True, False, False, False, False),
        ("1", True, True, True, False, False),
        ("1", True, True, False, True, False),
    ],
    ids=["on", "off", "no-alt", "cpu", "graph-warmup", "capture"],
)
def test_zeros_beside_gather_gate(monkeypatch, on, alt, cuda, graph, capturing, stream):
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_TAIL_ZEROS_BESIDE_GATHER", on)
    monkeypatch.setattr(
        kimi, "get_forward_context", lambda: SimpleNamespace(in_hipgraph=graph)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    second = _Stream("alt") if alt else None
    layer = SimpleNamespace(block_sparse_moe=SimpleNamespace(alt_stream=second))
    got = kimi.KimiDecoderLayer._zero_addends_stream(
        layer, SimpleNamespace(is_cuda=cuda)
    )
    assert got is (second if stream else None)


def test_pad_switch_is_on_unless_0(monkeypatch):
    monkeypatch.delenv("ATOM_ROW_SHARD_PAD_IN_PLACE", raising=False)
    assert kimi.envs.ATOM_ROW_SHARD_PAD_IN_PLACE
    monkeypatch.setenv("ATOM_ROW_SHARD_PAD_IN_PLACE", "0")
    assert not kimi.envs.ATOM_ROW_SHARD_PAD_IN_PLACE


def test_resident_residual_is_off_unless_1(monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_SHARD_RESIDUAL", raising=False)
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_RESIDUAL is False
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_RESIDUAL", "1")
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_RESIDUAL is True


@pytest.mark.parametrize("tokens", [13, 16])
def test_owned_rows_are_the_real_rows_of_the_shard(monkeypatch, tokens):
    monkeypatch.setattr(sequence_parallel, "get_tp_group", lambda: None)
    x = torch.arange(tokens * 2).view(tokens, 2)
    seen = []
    for rank in range(WORLD):
        rows = sequence_parallel.SequenceParallel(tokens, WORLD, rank)
        owned = rows.owned(x)
        assert owned.shape[0] == max(0, min(rows.chunk, tokens - rows.start))
        assert owned.shape[0] == 0 or owned.data_ptr() == x[rows.start].data_ptr()
        seen.append(owned)
    assert torch.equal(torch.cat(seen), x)


def _attn_res_rows(prefix, bank, score_weight, eps, norm, out_eps, blocks, out, pref):
    """`apply_attn_res_rows` as the torch reference over two zero addends."""
    _events.log.append(("attn_res_rows", prefix.shape[0]))
    zeros = torch.zeros_like(prefix)
    y, summed = _torch_attn_res(
        prefix, bank, score_weight, eps, zeros, norm, out_eps, zeros, None, blocks
    )
    out.copy_(y)
    pref.copy_(summed)


@pytest.fixture
def resident(logged, monkeypatch, stub_block_parts):
    """Two KDA layers and the output mix per rank; `link` keeps each row
    tail's residual on its rows as `_init_resident_rows` would."""
    empty = torch.Tensor.new_empty

    def poisoned_new_empty(self, *args, **kwargs):
        out = empty(self, *args, **kwargs)
        return out.fill_(float("nan")) if out.is_floating_point() else out

    monkeypatch.setattr(torch.Tensor, "new_empty", poisoned_new_empty)
    monkeypatch.setattr(kimi_k3_ops, "apply_attn_res_rows", _attn_res_rows)
    for name in ("kimi_k3_resident_attn_res", "kimi_k3_resident_aux"):
        monkeypatch.setattr(
            torch.ops.aiter, name, getattr(module_dispatch_ops, name), raising=False
        )
    context = stub_block_parts.compilation_config.static_forward_context

    def build(first, attention=None):
        stack = []
        for layer_num in (first, first + 1):
            layers = _layers(layer_num, attention)
            for layer in layers:
                moe = layer.block_sparse_moe
                moe.prefix = f"{layer.layer_name}.mlp"
                context[moe.prefix] = moe
            stack.append(layers)
        gen = torch.Generator().manual_seed(first)
        score_weight = torch.randn(HIDDEN, generator=gen)
        norm_weight = (1 + 0.1 * torch.randn(HIDDEN, generator=gen)).bfloat16()
        outputs = []
        for _ in range(WORLD):
            out_norm = _Stub()
            out_norm.weight = norm_weight
            site = kimi.AttnRes(_Stub(), _Stub(), out_norm=out_norm)
            site.score_weight = score_weight
            outputs.append(site)
        return [(*layers, site) for *layers, site in zip(*stack, outputs)]

    def link(model, on=True):
        for first, second, site in model:
            pairs = (
                (first.block_sparse_moe, second.self_attention_attn_res),
                (second.block_sparse_moe, site),
            )
            for moe, next_site in pairs:
                moe.resident_rows = on
                next_site.resident_from = moe.prefix if on else ""

    def forward(model, inputs):
        def on_rank(rank):
            first, second, site = model[rank]
            x, bank, add, add2 = (t.clone() for t in inputs)
            out = first(None, x, bank, add, add2)
            aggregate = first.aggregate_hidden_state(out, second)
            prefix, add, add2, bank = out
            mixed = second.self_attention_attn_res(prefix, bank, add, add2)
            aux = kimi._aux_from_next_residual(
                ((prefix, bank, add, add2), mixed), second.self_attention_attn_res
            )
            out = second(None, prefix, bank, add, add2)
            hidden, _ = site(out[0], out[3], out[1], out[2])
            return hidden, aggregate, aux, out

        try:
            return logged(on_rank)
        finally:
            attention_residual._ZERO_ADDENDS.clear()

    return SimpleNamespace(build=build, link=link, forward=forward)


def _owned(tokens, rank, x):
    chunk = -(-tokens // WORLD)
    return x[rank * chunk : (rank + 1) * chunk]


@pytest.mark.parametrize("first", [BLOCK - 1, BLOCK], ids=["mix-closes", "mid-block"])
@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_resident_rows_keep_the_stack_bit_for_bit(resident, monkeypatch, tokens, first):
    """A row tail that keeps its residual on its rows changes no output bit:
    not the next layer's, the output mix's or either draft tap's. Its gather
    goes; each mix after it runs on this rank's rows and gathers its output."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    model = resident.build(first)
    inputs = _inputs(tokens)
    gathered = resident.forward(model, inputs)
    resident.link(model)
    kept = resident.forward(model, inputs)

    chunk = -(-tokens // WORLD)
    mixes = ("all_gather", "attn_res", "attn_res_rows")
    for rank, ((want, want_log), (got, log)) in enumerate(zip(gathered, kept)):
        hidden, aggregate, aux, (residual, _, _, bank) = got
        assert torch.equal(hidden, want[0]) and not hidden.isnan().any()
        assert torch.equal(aggregate, want[1]) and torch.equal(aux, want[2])
        for kept_rows, want_rows in ((residual, want[3][0]), (bank, want[3][3])):
            assert torch.equal(
                _owned(tokens, rank, kept_rows), _owned(tokens, rank, want_rows)
            )
        assert [e for e in log if e[0] not in mixes] == [
            e for e in want_log if e[0] not in mixes
        ]
        # Full mixes: the first layer's input, the aggregate tap, the second
        # layer's input (once for the plain tap, once in its forward) and the
        # output. All but the first run on rows.
        real = max(0, min(chunk, tokens - rank * chunk))
        assert want_log.count(("attn_res", tokens)) == 5
        assert log.count(("attn_res", tokens)) == 1
        assert log.count(("attn_res_rows", real)) == (4 if real else 0)
        # Gathered: two layers' MoE inputs and residuals, or their MoE inputs,
        # four row mixes and the plain tap's prefix rows.
        assert want_log.count(("all_gather", chunk)) == 4
        assert log.count(("all_gather", chunk)) == 7


@pytest.mark.parametrize("tokens", [13, 1027])
def test_resident_mix_without_gather_keeps_its_rows(
    resident, logged, monkeypatch, tokens
):
    """Without `gather` a resident mix hands back this rank's rows of the
    gathered mix and writes no other row."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    model = resident.build(BLOCK)
    resident.link(model)
    inputs = _inputs(tokens)
    chunk = -(-tokens // WORLD)

    def on_rank(rank):
        first, second, _ = model[rank]
        x, bank, add, add2 = (t.clone() for t in inputs)
        prefix, add, add2, bank = first(None, x, bank, add, add2)
        site = second.self_attention_attn_res
        args = (prefix, bank, site.score_weight, site.eps, add)
        args += (site.out_norm.weight, site.out_eps, add2, site.num_blocks)
        moe = first.block_sparse_moe
        return moe.resident_attn_res(*args), moe.resident_attn_res(*args, gather=False)

    try:
        results = logged(on_rank)
    finally:
        attention_residual._ZERO_ADDENDS.clear()
    for rank, (((mixed, prefix), (kept, kept_prefix)), log) in enumerate(results):
        assert kept.shape == mixed.shape
        assert torch.equal(_owned(tokens, rank, kept), _owned(tokens, rank, mixed))
        assert torch.equal(
            _owned(tokens, rank, kept_prefix), _owned(tokens, rank, prefix)
        )
        others = torch.ones(tokens, dtype=torch.bool)
        others[rank * chunk : (rank + 1) * chunk] = False
        assert kept[others].isnan().all()
        # The layer's MoE input and the gathered mix; no residual, no second mix.
        assert log.count(("all_gather", chunk)) == 2


@pytest.mark.parametrize(
    "threshold,capturing,tbo_step",
    [("13", False, False), ("4", True, False), ("4", False, True)],
    ids=["at-threshold", "graph-capture", "tbo-step"],
)
def test_resident_rows_leave_other_steps_alone(
    resident, monkeypatch, threshold, capturing, tbo_step
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", threshold)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(ubatching, "tbo_active", lambda: tbo_step)
    model = resident.build(BLOCK - 1)
    inputs = _inputs(13)
    base = resident.forward(model, inputs)
    resident.link(model)
    for (want, want_log), (got, log) in zip(base, resident.forward(model, inputs)):
        assert log == want_log
        assert all(torch.equal(a, b) for a, b in zip(got[:3], want[:3]))
        assert all(torch.equal(a, b) for a, b in zip(got[3], want[3]))


def _resident_model(monkeypatch, layout, out_quant=None):
    """`layout`: per layer, (moe, post-attention op, input-mix quant dtype)."""
    from atom import plugin

    monkeypatch.setattr(plugin, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(kimi, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    context = {}

    def site(layer_idx, quant=None, block_size=BLOCK):
        return SimpleNamespace(
            enabled=True,
            block_size=block_size,
            layer_idx=layer_idx,
            num_blocks=None if block_size is None else -(-layer_idx // block_size),
            out_quant_dtype=quant,
            resident_from="",
        )

    layers = []
    for idx, (has_moe, op, quant) in enumerate(layout):
        layer = SimpleNamespace(
            layer_idx=idx,
            post_attention_op=op,
            self_attention_attn_res=site(idx, quant),
        )
        if has_moe:
            layer.block_sparse_moe = SimpleNamespace(
                prefix=f"layers.{idx}.mlp", resident_rows=False
            )
            context[layer.block_sparse_moe.prefix] = layer.block_sparse_moe
        layers.append(layer)
    model = SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=len(layers),
        output_attn_res=site(0, out_quant, None),
    )
    config = SimpleNamespace(
        enable_tbo=False,
        compilation_config=SimpleNamespace(static_forward_context=context),
    )
    return model, config


def _linked(model):
    return [
        getattr(
            layer, "block_sparse_moe", SimpleNamespace(resident_rows=False)
        ).resident_rows
        for layer in model.layers
    ]


MOE, DENSE, FP8 = (True, True, None), (False, False, None), torch.float8_e4m3fn


@pytest.mark.parametrize(
    "layout,out_quant,expected",
    [
        ([DENSE, MOE, MOE, MOE, MOE, MOE], None, [False] + [True] * 5),
        # Layer 4's mix banks its rows, so every mix after it must link.
        (
            [DENSE, MOE, MOE, MOE, MOE, MOE],
            FP8,
            [False, True, True, False, True, False],
        ),
        (
            [DENSE, MOE, MOE, MOE, MOE, (True, True, FP8)],
            None,
            [False, True, True, False, False, True],
        ),
        # A mix ahead of layer 4 banks nothing: later links stand alone.
        (
            [DENSE, MOE, (True, True, FP8), MOE, MOE, MOE],
            None,
            [False, False] + [True] * 4,
        ),
        # The mix ahead of a layer without the row tail would hand it rows.
        ([DENSE, MOE, (True, False, None), MOE], None, [False, False, False, True]),
        ([DENSE, MOE, DENSE, MOE], None, [False, False, False, True]),
    ],
    ids=[
        "all",
        "fp8-output",
        "fp8-after-close",
        "fp8-before-close",
        "no-op-next",
        "dense-next",
    ],
)
def test_resident_rows_link_only_where_every_reader_reads_rows(
    monkeypatch, layout, out_quant, expected
):
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_RESIDUAL", "1")
    model, config = _resident_model(monkeypatch, layout, out_quant)
    kimi.KimiLinearModel._init_resident_rows(model, config)
    assert _linked(model) == expected
    sites = [layer.self_attention_attn_res for layer in model.layers[1:]]
    for layer, site in zip(model.layers, [*sites, model.output_attn_res]):
        moe = getattr(layer, "block_sparse_moe", None)
        assert site.resident_from == (moe.prefix if moe and moe.resident_rows else "")


@pytest.mark.parametrize("case", ["off", "tbo", "pp", "plugin"])
def test_resident_rows_link_nothing_outside_one_eager_stage(monkeypatch, case):
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_RESIDUAL", "0" if case == "off" else "1")
    model, config = _resident_model(monkeypatch, [DENSE, MOE, MOE])
    if case == "tbo":
        config.enable_tbo = True
    elif case == "pp":
        monkeypatch.setattr(kimi, "get_pp_group", lambda: SimpleNamespace(world_size=2))
    elif case == "plugin":
        from atom import plugin

        monkeypatch.setattr(plugin, "is_plugin_mode", lambda: True)
    kimi.KimiLinearModel._init_resident_rows(model, config)
    assert _linked(model) == [False] * 3


# ---- ATOM_KIMI_K3_O_PROJ_PADDED_OUT (PKDA S2b): o_proj writes its partial
# into the padded buffer the row tail's reduce-scatter takes.


class _OProj:
    """RowParallelLinear's call surface: this rank's share of the mixer
    output, all-reduced unless the layer turns `reduce_results` off, and
    written into `out` when given."""

    runs_bf16 = False
    output_size = HIDDEN

    def __init__(self, attn) -> None:
        self.attn = attn
        self.reduce_results = True
        self.out_ok = True
        self.calls = []

    def supports_out(self):
        return self.out_ok

    def __call__(self, x, x_scale=None, out=None):
        assert x_scale is None
        self.calls.append(out is not None)
        y = (self.attn.share * x).to(x.dtype)
        if self.reduce_results:
            y = kimi.tensor_model_parallel_all_reduce(y)
        return y if out is None else out.copy_(y)


class _PaddedAttention(_Attention):
    """`_Attention`, its o_proj called as the KDA mixer calls it."""

    padded_out_rows = None  # as on KimiKDAAttention / KimiFullAttention
    o_proj_layer_name = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.o_proj = _OProj(self)

    def forward(self, x):
        _events.log.append(("attention", x.shape[0]))
        return kimi._o_proj_into_rows(self, torch.tanh(x @ self.w))


class _RankSlot:
    """`kimi._O_PROJ_ROWS` for each simulated rank: the ranks are threads
    here, processes in the engine."""

    def __init__(self) -> None:
        self.local = threading.local()

    def _items(self):
        if not hasattr(self.local, "items"):
            self.local.items = []
        return self.local.items

    def __bool__(self):
        return bool(self._items())

    def __len__(self):
        return len(self._items())

    def __setitem__(self, key, value):
        self._items()[key] = value

    def pop(self):
        return self._items().pop()

    def clear(self):
        self._items().clear()


@pytest.fixture
def o_proj_slot(monkeypatch):
    slot = _RankSlot()
    monkeypatch.setattr(kimi, "_O_PROJ_ROWS", slot)
    return slot


def _padded_layers(monkeypatch, layer_num, kda=True):
    monkeypatch.setenv("ATOM_KIMI_K3_O_PROJ_PADDED_OUT", "1")
    layers = _layers(layer_num, _PaddedAttention, kda)
    monkeypatch.delenv("ATOM_KIMI_K3_O_PROJ_PADDED_OUT")
    return layers


def _set_padded_out(layers, on):
    for layer in layers:
        attn = layer.self_attn
        attn.padded_out_rows = layer.block_sparse_moe.padded_out_rows if on else None


def test_o_proj_padded_out_is_off_unless_1(monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_K3_O_PROJ_PADDED_OUT", raising=False)
    from atom.utils import envs

    assert envs.ATOM_KIMI_K3_O_PROJ_PADDED_OUT is False
    for layer in _layers(4, _PaddedAttention):
        assert layer.self_attn.padded_out_rows is None
        assert layer.self_attn.o_proj_layer_name is None


@pytest.mark.parametrize("kda", [True, False], ids=["kda", "mla"])
def test_o_proj_padded_out_wires_each_row_tail_layer(monkeypatch, kda):
    for layer in _padded_layers(monkeypatch, 4, kda):
        moe = layer.block_sparse_moe
        assert layer.post_attention_op
        assert layer.self_attn.padded_out_rows == moe.padded_out_rows
        # The MLA attention is traced: it reaches o_proj through the op.
        want = None if kda else layer.layer_name
        assert layer.self_attn.o_proj_layer_name == want
        assert moe not in list(layer.self_attn.modules())


@pytest.mark.parametrize("layer_num", [4, 5], ids=["block-closes", "mid-block"])
@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_o_proj_padded_out_keeps_the_row_tail_bit_for_bit(
    logged, monkeypatch, o_proj_slot, tokens, layer_num
):
    """On a step with pad rows o_proj writes into the padded buffer that the
    attention reduce-scatter then takes as it is, as the routed and shared
    partials are; without pad rows nothing changes. Outputs, collectives and
    kernels stay the same."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    layers = _padded_layers(monkeypatch, layer_num)
    inputs = _inputs(tokens)
    pad_rows, local = sequence_parallel._pad_rows, threading.local()

    def recorded(x, missing):
        out = pad_rows(x, missing)
        if x.shape[0] == tokens:  # a reduce-scatter's input, not a shard
            local.pads.append((x.shape[-1], out is getattr(x, "_padded_rows_of", 0)))
        return out

    monkeypatch.setattr(sequence_parallel, "_pad_rows", recorded)

    def forward(rank):
        local.pads = []
        out = layers[rank](None, *(x.clone() for x in inputs))
        return (
            out,
            local.pads,
            list(layers[rank].self_attn.o_proj.calls),
            len(o_proj_slot),
        )

    _set_padded_out(layers, False)
    base = logged(forward)
    for layer in layers:
        layer.self_attn.o_proj.calls.clear()
    _set_padded_out(layers, True)
    got = logged(forward)

    padded = tokens % WORLD != 0
    for (want, want_log), (have, log) in zip(base, got):
        (want_out, want_pads, want_calls, _), (out, pads, calls, left) = want, have
        assert log == want_log
        assert all(map(torch.equal, _bits(out), _bits(want_out)))
        assert want_pads == [(HIDDEN, False), (LATENT, padded), (HIDDEN, padded)]
        assert pads == [(HIDDEN, padded), (LATENT, padded), (HIDDEN, padded)]
        assert want_calls == [False] and calls == [padded]
        assert left == 0


@pytest.mark.parametrize("first", [BLOCK - 1, BLOCK], ids=["mix-closes", "mid-block"])
@pytest.mark.parametrize("tokens", [13, 1027])
def test_o_proj_padded_out_keeps_the_resident_stack_bit_for_bit(
    resident, monkeypatch, o_proj_slot, tokens, first
):
    """Under ATOM_KIMI_K3_ROW_SHARD_RESIDUAL as well: S2b on against off, with
    both row tails keeping their residual on their rows."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    monkeypatch.setenv("ATOM_KIMI_K3_O_PROJ_PADDED_OUT", "1")
    model = resident.build(first, _PaddedAttention)
    monkeypatch.delenv("ATOM_KIMI_K3_O_PROJ_PADDED_OUT")
    resident.link(model)
    layers = [
        layer for first_layer, second, _ in model for layer in (first_layer, second)
    ]
    _set_padded_out(layers, False)
    base = resident.forward(model, _inputs(tokens))
    _set_padded_out(layers, True)
    got = resident.forward(model, _inputs(tokens))
    for rank, ((want, want_log), (have, log)) in enumerate(zip(base, got)):
        hidden, aggregate, aux, (residual, _, _, bank) = have
        assert log == want_log
        assert torch.equal(hidden, want[0]) and not hidden.isnan().any()
        assert torch.equal(aggregate, want[1]) and torch.equal(aux, want[2])
        for kept_rows, want_rows in ((residual, want[3][0]), (bank, want[3][3])):
            assert torch.equal(
                _owned(tokens, rank, kept_rows), _owned(tokens, rank, want_rows)
            )
    for first_layer, second, _ in model:
        for layer in (first_layer, second):
            assert layer.self_attn.o_proj.calls == [False, True]


def _one_layer_rows(monkeypatch, tokens, case):
    """Per rank: (o_proj's result, whether it went to a padded buffer, what
    the slot holds afterwards) for one call of `_o_proj_into_rows` in `case`."""
    layers = _padded_layers(monkeypatch, 4)
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    if case == "off":
        _set_padded_out(layers, False)
    elif case == "below-floor":
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", str(tokens + 1))
    elif case == "graph":
        _in_graph(monkeypatch, token_graph=False)
    elif case == "no-out":
        for layer in layers:
            layer.self_attn.o_proj.out_ok = False
    elif case == "runs-bf16":
        for layer in layers:
            layer.self_attn.o_proj.runs_bf16 = True
    elif case == "pad-in-place-off":
        monkeypatch.setenv("ATOM_ROW_SHARD_PAD_IN_PLACE", "0")
    x = torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(tokens))
    return layers, x.bfloat16()


@pytest.mark.parametrize(
    "case,tokens",
    [
        ("on", 13),
        ("on", 16),
        ("off", 13),
        ("below-floor", 13),
        ("graph", 13),
        ("no-out", 13),
        ("runs-bf16", 13),
        ("pad-in-place-off", 13),
    ],
)
def test_o_proj_rows_gate(ranks, monkeypatch, o_proj_slot, case, tokens):
    """Only an eager row tail with pad rows, an o_proj that writes `out=` and
    in-place padding put the partial into a padded buffer; everywhere else
    o_proj runs as before and nothing is registered."""
    layers, x = _one_layer_rows(monkeypatch, tokens, case)
    want_padded = case == "on" and tokens % WORLD != 0

    def on_rank(rank):
        attn = layers[rank].self_attn
        attn.o_proj.reduce_results = False
        y = kimi._o_proj_into_rows(attn, x)
        buffer = getattr(y, "_padded_rows_of", None)
        held = list(o_proj_slot._items())
        o_proj_slot.clear()
        return y, buffer, held, attn.o_proj.calls

    for rank, (y, buffer, held, calls) in enumerate(ranks(on_rank)):
        share = layers[rank].self_attn.share
        assert torch.equal(y, (share * x).to(x.dtype))
        assert calls == [want_padded]
        if want_padded:
            assert held == [y] and buffer.shape == (16, HIDDEN)
            assert buffer.data_ptr() == y.data_ptr() and not buffer[tokens:].any()
        else:
            assert held == [] and buffer is None


def test_o_proj_rows_taken_only_as_themselves(monkeypatch, ranks, o_proj_slot):
    """The post-attention op takes the registered rows only when it is handed
    them; a copy (as a compiler might make) pads as before. The slot is
    emptied either way, and an empty slot hands its input back."""
    layers, x = _one_layer_rows(monkeypatch, 13, "on")

    def on_rank(rank):
        attn = layers[rank].self_attn
        attn.o_proj.reduce_results = False
        y = kimi._o_proj_into_rows(attn, x)
        copy = y.clone()
        missed = kimi._take_o_proj_rows(copy)
        after_miss = len(o_proj_slot)
        y2 = kimi._o_proj_into_rows(attn, x)
        view = y2.view(13, HIDDEN)
        taken = kimi._take_o_proj_rows(view)
        after_take = len(o_proj_slot)
        again = kimi._take_o_proj_rows(view)
        return copy, missed, after_miss, y2, taken, after_take, again, view

    for copy, missed, after_miss, y2, taken, after_take, again, view in ranks(on_rank):
        assert missed is copy and after_miss == 0
        assert taken is y2 and after_take == 0
        assert again is view


def test_mla_o_proj_runs_through_the_op(
    monkeypatch, ranks, o_proj_slot, stub_block_parts
):
    """A traced (MLA) attention reaches o_proj through `kimi_k3_o_proj_rows`,
    which looks its layer up and runs `_o_proj_into_rows` on its attention;
    its fake has o_proj's output shape."""
    layers = _padded_layers(monkeypatch, 4, kda=False)
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    x = torch.randn(13, HIDDEN, generator=torch.Generator().manual_seed(5)).bfloat16()
    core = torch.tanh(x @ layers[0].self_attn.w)
    fake = kimi._kimi_k3_o_proj_rows_fake(core, None, "unused", HIDDEN)
    assert fake.shape == (13, HIDDEN) and fake.dtype == torch.bfloat16

    def on_rank(rank):
        layer = layers[rank]
        attn = layer.self_attn
        attn.o_proj.reduce_results = False
        attn._gated_core = lambda positions, h: (torch.tanh(h @ attn.w), None)
        y = kimi.KimiFullAttention.forward(attn, None, x)
        held = list(o_proj_slot._items())
        o_proj_slot.clear()
        attn.o_proj_layer_name = None
        plain = kimi.KimiFullAttention.forward(attn, None, x)
        return y, held, plain, layer

    context = stub_block_parts.compilation_config.static_forward_context
    for y, held, plain, layer in ranks(on_rank):
        assert context[layer.layer_name] is layer
        assert held == [y] and y._padded_rows_of.shape == (16, HIDDEN)
        assert torch.equal(_bits([y])[0], _bits([plain])[0])
        assert layer.self_attn.o_proj.calls == [True, False]


# ---- ATOM_KIMI_K3_ROW_TAIL_OWNED_MIX (PKDA S2c): a rank whose rows run past
# T mixes its real rows in place instead of padding its prefix and bank rows.


def _attn_res_into(
    prefix, bank, score_weight, eps, add, norm, out_eps, blocks, out, pref
):
    """`apply_attn_res_into` as the torch reference (row by row, so a slice of
    rows gives the rows of the full mix)."""
    _events.log.append(("attn_res_into", prefix.shape[0]))
    assert bank.stride(-1) == 1 and out.is_contiguous()
    y, summed = _torch_attn_res(
        prefix, bank, score_weight, eps, add, norm, out_eps, None, None, blocks
    )
    out.copy_(y)
    if add is None:
        assert pref is None
    else:
        pref.copy_(summed)


@pytest.fixture
def owned_mix(logged, monkeypatch):
    monkeypatch.setattr(kimi_k3_ops, "apply_attn_res_into", _attn_res_into)
    pad_rows, local = sequence_parallel._pad_rows, threading.local()

    def recorded(x, missing):
        if missing and hasattr(local, "shard_pads"):
            local.shard_pads.append((x.dim(), x.shape[0] + missing))
        return pad_rows(x, missing)

    monkeypatch.setattr(sequence_parallel, "_pad_rows", recorded)
    return local


def _owned_layers(monkeypatch, layer_num, attention=None):
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_TAIL_OWNED_MIX", "1")
    layers = _layers(layer_num, attention)
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_TAIL_OWNED_MIX")
    return layers


def _real_rows(tokens, rank):
    chunk = -(-tokens // WORLD)
    return chunk, max(0, min(chunk, tokens - rank * chunk))


def test_owned_mix_is_off_unless_1(monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_TAIL_OWNED_MIX", raising=False)
    from atom.utils import envs

    assert envs.ATOM_KIMI_K3_ROW_TAIL_OWNED_MIX is False
    assert not any(layer.owned_mix for layer in _layers(4))
    assert all(layer.owned_mix for layer in _owned_layers(monkeypatch, 4))


@pytest.mark.parametrize("layer_num", [4, 5], ids=["block-closes", "mid-block"])
@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_owned_mix_keeps_the_row_tail_bit_for_bit(
    logged, owned_mix, monkeypatch, tokens, layer_num
):
    """The short ranks mix only their real rows, read in place: no prefix or
    window shard is padded, every output bit stays, and every other rank (and
    every step without pad rows) runs exactly as before."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    layers = _owned_layers(monkeypatch, layer_num)
    inputs = _inputs(tokens)

    def forward(rank):
        owned_mix.shard_pads = []
        out = layers[rank](None, *(x.clone() for x in inputs))
        return out, owned_mix.shard_pads

    for layer in layers:
        layer.owned_mix = False
    base = logged(forward)
    for layer in layers:
        layer.owned_mix = True
    got = logged(forward)

    closes = layer_num % BLOCK == 0
    for rank, (((want, want_pads), want_log), ((out, pads), log)) in enumerate(
        zip(base, got)
    ):
        chunk, real = _real_rows(tokens, rank)
        # Shards pad to a rank's row count, reduce-scatter inputs to all rows.
        want_pads = [d for d, n in want_pads if n == chunk]
        pads = [d for d, n in pads if n == chunk]
        assert all(map(torch.equal, _bits(out), _bits(want)))
        assert not out[0].isnan().any()
        if real == chunk:
            assert log == want_log and pads == want_pads
            continue
        # Off: the short rank padded its window (and prefix, mid-block) shards.
        assert want_pads == ([3] if closes else [2, 3])
        assert pads == []
        mix = ("attn_res", chunk)
        swapped = [("attn_res_into", real)] if real else []
        i = want_log.index(mix, 1)  # the first mix is the input site's
        assert log == want_log[:i] + swapped + want_log[i + 1 :]


@pytest.mark.parametrize("first", [BLOCK - 1, BLOCK], ids=["mix-closes", "mid-block"])
@pytest.mark.parametrize("tokens", [13, 1027])
def test_owned_mix_keeps_the_resident_stack_bit_for_bit(
    resident, owned_mix, o_proj_slot, monkeypatch, tokens, first
):
    """With S2's resident rows and S2b's padded o_proj as well: all three on
    against S2 alone, bit for bit."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_TAIL_OWNED_MIX", "1")
    monkeypatch.setenv("ATOM_KIMI_K3_O_PROJ_PADDED_OUT", "1")
    model = resident.build(first, _PaddedAttention)
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_TAIL_OWNED_MIX")
    monkeypatch.delenv("ATOM_KIMI_K3_O_PROJ_PADDED_OUT")
    resident.link(model)
    layers = [layer for a, b, _ in model for layer in (a, b)]

    def switch(on):
        _set_padded_out(layers, on)
        for layer in layers:
            layer.owned_mix = on

    switch(False)
    base = resident.forward(model, _inputs(tokens))
    switch(True)
    got = resident.forward(model, _inputs(tokens))
    for rank, ((want, _), (have, log)) in enumerate(zip(base, got)):
        hidden, aggregate, aux, (residual, _, _, bank) = have
        assert torch.equal(hidden, want[0]) and not hidden.isnan().any()
        assert torch.equal(aggregate, want[1]) and torch.equal(aux, want[2])
        for kept_rows, want_rows in ((residual, want[3][0]), (bank, want[3][3])):
            assert torch.equal(
                _owned(tokens, rank, kept_rows), _owned(tokens, rank, want_rows)
            )
        chunk, real = _real_rows(tokens, rank)
        assert log.count(("attn_res_into", real)) == (2 if 0 < real < chunk else 0)


@pytest.mark.parametrize(
    "case", ["disabled", "no-window", "fp8-out", "fp32-rows", "resident"]
)
def test_owned_mix_keeps_other_mixes_as_they_are(logged, owned_mix, monkeypatch, case):
    """A mix that is not the plain BF16 kernel with one addend (residuals off,
    no candidates, a fused quant, other dtypes, a resident link) runs as
    before, padded shards included."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    layers = _owned_layers(monkeypatch, 5)
    rank = WORLD - 1
    layer = layers[rank]
    site = layer.mlp_attn_res
    x, bank, *_ = _inputs(13)
    window = site.window(bank)
    prefix = x
    if case == "disabled":
        monkeypatch.setattr(site, "enabled", False)
    elif case == "no-window":
        window = None
    elif case == "fp8-out":
        monkeypatch.setattr(
            type(site), "out_quant_dtype", property(lambda self: torch.float8_e4m3fn)
        )
    elif case == "fp32-rows":
        prefix = x.float()
    elif case == "resident":
        site.resident_from = "somewhere"
    assert not layer._owned_mix_fits(prefix, window, x)
    assert layer._owned_mix_fits(x, site.window(bank), x) == (
        case not in ("disabled", "fp8-out", "resident")
    )
