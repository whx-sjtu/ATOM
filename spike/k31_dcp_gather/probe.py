"""One leased rank's fixed data/scale gather pairs; imports are CPU-safe."""

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path


class ByteMismatch(RuntimeError):
    pass


def shape(keys, plane):
    return (keys // 8, 4, 16) if plane == 0 else (keys // 8, 4)


def pattern(keys, rank, plane, repetition):
    import numpy as np

    dimensions = shape(keys, plane)
    count = int(np.prod(dimensions))
    values = np.arange(count, dtype=np.uint32)
    values += rank * 31 + plane * 67 + repetition * 113 + keys // 8192 * 19
    return (values % 251).astype(np.uint8).reshape(dimensions)


def reference(keys, plane, repetition):
    import numpy as np

    return np.concatenate([pattern(keys, r, plane, repetition) for r in range(8)])


def verify(actual, expected):
    import numpy as np

    if actual.dtype != np.uint8 or actual.shape != expected.shape:
        raise ByteMismatch(f"shape/dtype: {actual.shape}/{actual.dtype}")
    unequal = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
    if unequal.size:
        raise ByteMismatch(f"{unequal.size} unequal bytes; first index {unequal[0]}")
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "bytes_checked": actual.nbytes,
        "sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "equal": True,
    }


def self_check():
    import numpy as np

    checked = 0
    for keys in (393216, 401408):
        for repetition in range(3):
            for plane in (0, 1):
                expected = reference(keys, plane, repetition)
                verify(expected.copy(), expected)
                corrupted = expected.copy()
                corrupted.reshape(-1)[-1] ^= 1
                try:
                    verify(corrupted, expected)
                except ByteMismatch:
                    checked += 1
                else:
                    raise AssertionError("last-byte corruption was accepted")
                if repetition:
                    assert not np.array_equal(expected, reference(keys, plane, 0))
    assert not any(name in sys.modules for name in ("torch", "aiter", "atom"))
    return {"full_plane_checks": checked, "last_byte_mismatches_rejected": checked}


def worker(plan_path, rank):
    from run import STOP_SIGNALS, load_frozen, save

    signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)

    frozen = load_frozen(plan_path, require_grant=True)
    protocol = frozen["protocol"]
    if rank not in range(8) or int(os.environ.get("LOCAL_RANK", "-1")) != rank:
        raise ValueError("rank/local device mismatch")
    out = Path(frozen["output"])
    results = []
    state = {"rank": rank, "pid": os.getpid(), "results": results}

    def mark(phase, **values):
        state.update(phase=phase, monotonic=time.monotonic(), **values)
        save(out / f"rank-{rank}.json", state)
        with (out / f"rank-{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(state, results=list(results))) + "\n")

    mark("initializing")
    try:
        import aiter
        import torch
        from aiter.dist import parallel_state as ps

        import atom
        from atom.distributed.dcp_utils import get_dcp_group

        if (
            not Path(atom.__file__)
            .resolve()
            .is_relative_to(Path(frozen["root"]) / "atom")
        ):
            raise RuntimeError("unexpected ATOM import")
        if (
            not Path(aiter.__file__)
            .resolve()
            .is_relative_to(Path(protocol["aiter_root"]))
        ):
            raise RuntimeError("unexpected AITER import")
        torch.cuda.set_device(rank)
        if torch.cuda.device_count() != 8 or "MI355" not in torch.cuda.get_device_name(
            rank
        ):
            raise RuntimeError("expected eight MI355X local devices")
        ps.init_distributed_environment(world_size=8, rank=rank, local_rank=rank)
        ps.initialize_model_parallel(
            tensor_model_parallel_size=8,
            decode_context_model_parallel_size=8,
            pipeline_model_parallel_size=1,
            backend="nccl",
        )
        group, tp, dcp = get_dcp_group(), ps.get_tp_group(), ps.get_dcp_group()
        if (
            group is not tp
            or list(tp.ranks) != list(range(8))
            or list(dcp.ranks) != list(range(8))
        ):
            raise RuntimeError("unexpected DCP resolver/rank order")
        if group.device_group is ps.get_world_group().device_group:
            raise RuntimeError("world group is not the selected device group")
        if (
            group.rank_in_group != rank
            or torch.distributed.get_backend(group.device_group) != "nccl"
        ):
            raise RuntimeError("unexpected group rank/backend")
        state["runtime"] = {
            "atom": atom.__file__,
            "aiter": aiter.__file__,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "rccl": torch.cuda.nccl.version(),
            "group": group.unique_name,
            "selected_is_tp": group is tp,
            "selected_is_dcp": group is dcp,
            "tp_dcp_share_device_group": tp.device_group is dcp.device_group,
            "selected_device_group_id": id(group.device_group),
            "tp_device_group_id": id(tp.device_group),
            "dcp_device_group_id": id(dcp.device_group),
            "ranks": list(group.ranks),
            "device": str(group.device),
            "stream": torch.cuda.current_stream().cuda_stream,
            "rccl_libraries": sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if "librccl" in line and line.split()[-1].startswith("/")
                }
            ),
        }
        mark("ready")
        pairs = [(keys, rep) for keys in protocol["global_keys"] for rep in range(3)]
        for index, (keys, repetition) in enumerate(pairs):
            while not (out / f"go-{index}").exists():
                if (out / "abort").exists():
                    raise RuntimeError("controller aborted")
                time.sleep(0.01)
            mark("staging", pair=index, keys=keys, repetition=repetition)
            inputs = [
                torch.from_numpy(pattern(keys, rank, plane, repetition)).to(
                    device=torch.device("cuda", rank)
                )
                for plane in (0, 1)
            ]
            if not all(t.is_contiguous() and t.dtype == torch.uint8 for t in inputs):
                raise RuntimeError("expected contiguous uint8 inputs")
            current = torch.cuda.current_stream()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record(current)
            began = time.monotonic()
            mark("data_enqueue")
            data = group.all_gather(inputs[0], use_custom=False, dim=0)
            data_returned = time.monotonic()
            mark("scale_enqueue", data_host_return=data_returned)
            scales = group.all_gather(inputs[1], use_custom=False, dim=0)
            scales_returned = time.monotonic()
            end.record(current)
            mark("pair_enqueued", scales_host_return=scales_returned)
            while not end.query():
                if (out / "abort").exists():
                    raise RuntimeError("controller aborted")
                time.sleep(0.001)
            completed = time.monotonic()
            checks = [
                verify(tensor.cpu().numpy(), reference(keys, plane, repetition))
                for plane, tensor in enumerate((data, scales))
            ]
            results.append(
                {
                    "pair": index,
                    "keys": keys,
                    "repetition": repetition,
                    "host_data_s": data_returned - began,
                    "host_scale_s": scales_returned - data_returned,
                    "device_pair_ms": start.elapsed_time(end),
                    "completion_s": completed - began,
                    "planes": checks,
                }
            )
            mark("complete")
            del inputs, data, scales
        while not (out / "finish").exists():
            if (out / "abort").exists():
                raise RuntimeError("controller aborted")
            time.sleep(0.01)
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
        mark("finished")
    except BaseException as error:
        mark(
            "error",
            error=repr(error),
            mismatch=isinstance(error, ByteMismatch),
            traceback=traceback.format_exc(),
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
    elif args.worker is not None and args.rank is not None:
        worker(args.worker, args.rank)
    else:
        parser.error("use --self-check or --worker PLAN --rank RANK")


if __name__ == "__main__":
    main()
