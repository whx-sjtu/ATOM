# SPDX-License-Identifier: MIT
"""Kimi row-sharded residual and MoE tail on CPU-simulated TP ranks."""

import ast
import threading
from concurrent.futures import ThreadPoolExecutor
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
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", raising=False)
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_SHARD_GROUP_RS", raising=False)
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
    """Attention residual and output RMS norm in FP32 torch."""
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


def _layers(layer_num):
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
        kimi_kda_layers=[layer_num],
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
            patch.setattr(kimi, "KimiKDAAttention", _Attention)
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
        (8191, None, None, False),
        (8192, None, None, True),
        (8192, "8192", None, False),
        (8193, "8192", None, True),
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
        (None, True, 4096, False),
        ("1024", True, 1023, False),
        ("1024", True, 1024, True),
        ("1024", False, 4096, False),
    ],
    ids=["default-off", "below-floor", "at-floor", "other-capture"],
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
    monkeypatch.delenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS")
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


def test_pynccl_cap_and_group_are_off_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", raising=False)
    monkeypatch.delenv("ATOM_KIMI_K3_ROW_SHARD_GROUP_RS", raising=False)
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB == 0
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_GROUP_RS is False
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB", "48")
    monkeypatch.setenv("ATOM_KIMI_K3_ROW_SHARD_GROUP_RS", "1")
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_PYNCCL_RS_MIB == 48
    assert kimi.envs.ATOM_KIMI_K3_ROW_SHARD_GROUP_RS is True


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
