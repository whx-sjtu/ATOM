# SPDX-License-Identifier: MIT
"""TP row sharding shared by residual and model-tail paths."""

from collections.abc import Callable

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from torch import nn

from atom.utils import envs


def _pad_rows(x: torch.Tensor, missing: int) -> torch.Tensor:
    """`x` with `missing` zero rows appended on dim 0, for any rank.

    Under ``ATOM_ROW_SHARD_PAD_IN_PLACE`` only the appended rows are zeroed,
    and the rows of a `SequenceParallel.padded_rows` buffer are that buffer."""
    if not missing:
        return x
    if not envs.ATOM_ROW_SHARD_PAD_IN_PLACE:
        return nn.functional.pad(x, (0, 0) * (x.dim() - 1) + (0, missing))
    padded = getattr(x, "_padded_rows_of", None)
    if padded is not None and padded.shape[0] == x.shape[0] + missing:
        return padded
    out = x.new_empty((x.shape[0] + missing, *x.shape[1:]))
    out[: x.shape[0]].copy_(x)
    out[x.shape[0] :].zero_()
    return out


class _SideStream:
    """A priority -1 stream whose work starts after what the current stream
    has queued so far."""

    def __init__(self) -> None:
        self.stream = torch.cuda.Stream(priority=-1)

    def run(self, fn: Callable[[], None], *uses: torch.Tensor) -> Callable[[], None]:
        """Queue `fn` here; `uses` are current-stream tensors it reads. The
        returned callable makes the then-current stream wait for `fn`."""
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            fn()
            done = torch.cuda.Event()
            done.record()
        for x in uses:
            x.record_stream(self.stream)
        return lambda: torch.cuda.current_stream().wait_event(done)


_SIDE: list = []


def _side_comm():
    """A second pynccl communicator over the TP group, and its side stream.

    Built on first use, which every TP rank reaches on the same step (its
    construction is a collective), as `tbo_get_ubatch_tp_comm` does. Its own
    communicator keeps the side reduce out of the TP group's issue order."""
    if not _SIDE:
        from aiter.dist.device_communicators.communicator_pynccl import (
            PyNcclCommunicator,
        )

        tp = get_tp_group()
        _SIDE.append(
            (PyNcclCommunicator(group=tp.cpu_group, device=tp.device), _SideStream())
        )
    return _SIDE[0]


class SequenceParallel:
    """Shard rows across TP ranks, gather rows, or scatter partial sums.

    Pad to a multiple of the TP size and drop padded rows on gather.
    Per-token work runs locally; consumers needing every row gather first.
    ``custom`` runs both collectives on AITER's custom kernels where they fit
    and on pynccl elsewhere, either of which a graph can capture, instead of
    on RCCL through torch.distributed. With ``custom``, reduce-scatter inputs
    above ``pynccl_rs_bytes`` (0: none) go straight to pynccl, and
    ``group_rs`` lets `scatter_pair` issue two reduce-scatters as one NCCL
    group. ``side_rs`` (eager only) lets `scatter_side` reduce on a second
    communicator on a side stream.
    """

    def __init__(
        self,
        num_tokens: int,
        tp_size: int,
        tp_rank: int,
        custom: bool = False,
        pynccl_rs_bytes: int = 0,
        group_rs: bool = False,
        side_rs: bool = False,
    ) -> None:
        self.num_tokens = num_tokens
        self.tp_size = tp_size
        self.chunk = -(-num_tokens // tp_size)
        self.pad = self.chunk * tp_size - num_tokens
        self.start = tp_rank * self.chunk
        self.group = get_tp_group()
        self.custom = custom
        self.pynccl_rs_bytes = pynccl_rs_bytes if custom else 0
        self.group_rs = group_rs and custom
        self.side_rs = side_rs and custom

    def shard(self, x: torch.Tensor) -> torch.Tensor:
        """Replicated `[T, ...]` -> this rank's rows (a view unless padded).

        Only the local rows are padded, never the whole replicated tensor."""
        rows = x[self.start : self.start + self.chunk]
        return _pad_rows(rows, self.chunk - rows.shape[0])

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows -> all `[T, ...]` rows."""
        if self.custom:
            out = self.group.device_communicator.all_gather(x.contiguous(), 0)
        else:
            out = self.group.all_gather(x.contiguous(), use_custom=False, dim=0)
        return out[: self.num_tokens] if self.pad else out

    def scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Per-rank partial sums over all `[T, ...]` rows -> summed local rows."""
        return self._scatter_padded(_pad_rows(x.contiguous(), self.pad))

    def scatter_pair(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`scatter(a), scatter(b)`; under ``group_rs``, when `b` runs on
        pynccl, both as one NCCL group (so `a` runs on pynccl too)."""
        a, b = (_pad_rows(x.contiguous(), self.pad) for x in (a, b))
        if self.group_rs and self._on_pynccl(b):
            out_a, out_b = self._pynccl_scatter(a, b)
            return out_a, out_b
        return self._scatter_padded(a), self._scatter_padded(b)

    def scatter_side(self, x: torch.Tensor) -> Callable[[], torch.Tensor]:
        """`scatter(x)` on pynccl over a second communicator, on a side stream,
        after the work queued so far. Returns a callable that makes the current
        stream wait for the reduce and returns its rows. One side reduce at a
        time: the side stream runs them in issue order."""
        x = _pad_rows(x.contiguous(), self.pad)
        nccl, side = _side_comm()
        out = x.new_empty((x.shape[0] // self.tp_size, *x.shape[1:]))
        wait = side.run(lambda: nccl.reduce_scatter(out, x), x)

        def rows() -> torch.Tensor:
            wait()
            return out

        return rows

    def _scatter_padded(self, x: torch.Tensor) -> torch.Tensor:
        if self.pynccl_rs_bytes and self._on_pynccl(x):
            return self._pynccl_scatter(x)[0]
        return self.group.reduce_scatter_tensor(x, use_custom=self.custom, dim=0)

    def _on_pynccl(self, x: torch.Tensor) -> bool:
        """Whether reducing `x` belongs on pynccl: above ``pynccl_rs_bytes``,
        or where AITER's custom kernel would hand it to pynccl itself."""
        comm = self.group.device_communicator
        if comm.pynccl_comm is None or comm.pynccl_comm.disabled:
            return False
        if self.pynccl_rs_bytes and x.nbytes > self.pynccl_rs_bytes:
            return True
        ca = comm.ca_comm
        return ca is None or ca.disabled or not ca.should_custom_rs(x, 0)

    def _pynccl_scatter(self, *xs: torch.Tensor) -> list[torch.Tensor]:
        """Reduce-scatter each padded input on pynccl, as one group if several."""
        nccl = self.group.device_communicator.pynccl_comm
        outs = [x.new_empty((x.shape[0] // self.tp_size, *x.shape[1:])) for x in xs]
        if len(xs) == 1:
            nccl.reduce_scatter(outs[0], xs[0])
            return outs
        nccl.group_start()
        try:
            for out, x in zip(outs, xs):
                nccl.reduce_scatter(out, x)
        finally:
            nccl.group_end()
        return outs

    @classmethod
    def maybe(cls, num_tokens: int) -> "SequenceParallel | None":
        """Sharding for a forward of `num_tokens`, or None to stay replicated:
        no TP, below `ATOM_QWEN4_EXP_SP_MIN_TOKENS`, or under graph capture.
        The target model adds its own fused-quant floor
        (`Qwen4ExpModel._sequence_parallel`); the MTP drafter, whose unfused
        hyper-connections never quantize, uses this gate as is."""
        min_tokens = envs.ATOM_QWEN4_EXP_SP_MIN_TOKENS
        if (
            min_tokens <= 0
            or num_tokens < min_tokens
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return None
        return cls(num_tokens, tp_size, get_tensor_model_parallel_rank())

    def padded_rows(self, like: torch.Tensor) -> torch.Tensor | None:
        """Under ``ATOM_ROW_SHARD_PAD_IN_PLACE``, on a step with pad rows, an
        uninitialized `[T, *like.shape[1:]]` tensor for a producer to write the
        partials `scatter` reduces into: the leading rows of a buffer whose pad
        rows are zero, which `scatter` reduces without copying. Otherwise None.
        """
        if not self.pad or not envs.ATOM_ROW_SHARD_PAD_IN_PLACE:
            return None
        buffer = like.new_empty((self.num_tokens + self.pad, *like.shape[1:]))
        buffer[self.num_tokens :].zero_()
        rows = buffer[: self.num_tokens]
        rows._padded_rows_of = buffer
        return rows
