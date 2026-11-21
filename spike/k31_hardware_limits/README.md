# Per-device compute, memory and cache calibration

This spike records empirical rates on one named gfx950 device. It does not
alter inference or establish an exhaustive hardware ceiling. The copied
`kernel-perf/probe_roofline.py` is an external, SHA256-pinned input; its Triton
kernels, event timer, cache sweep and estimator are reused. The adapter does
not consume its unverified gfx950 throughput constant.

## Inputs and command

Use the reviewed `/opt/venv` ROCm runtime and a private writable cache/TMPDIR.
The source identities and binary hashes belong in the frozen external run
plan. The adapter checks the original probe hash and AITER commit, and rejects
tracked AITER changes. Preserve the exact source and command for every round.

```bash
/opt/venv/bin/python spike/k31_hardware_limits/probe.py \
  --probe /app/k31-agent-team-private/task-13/probe_roofline.py \
  --aiter /app/k31-agent-team-private/task-6/aiter-572232d \
  --bdf 0000:75:00.0 --mode roofline \
  --what compute,memory,cache --clock-seconds 12 --out "$RUN/roofline.json"
```

The other modes are `--mode a16w16` and `--mode a4w4`, each with its own fresh
output path. They execute the pinned AITER scripts through `runpy`, with these
original CLI arguments:

```text
op_tests/test_gemm_a16w16.py -d bf16 -o bf16 -mnk 8192,8192,8192
op_tests/test_gemm_a4w4.py -d bf16 -mnk 2048,8192,8192 -mnk_splitk
```

Set `PYTHONPATH` to the exact ATOM checkout followed by the pinned AITER root
and `/opt/rocm-7.2.4/share/amd_smi`. Set `HIP_VISIBLE_DEVICES=0` and
`ROCR_VISIBLE_DEVICES=0`; the adapter verifies that the sole visible device
has the reviewed BDF and gfx950 architecture. Use `AITER_LOG_TUNED_CONFIG=1`
and `AITER_LOG_MORE=0`; leave `AITER_SMI_MONITOR` disabled, because this adapter
records clocks during the original timing interval. Fix `PYTHONHASHSEED=0`,
`PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and one Torch/OMP/BLAS thread.
No compiler, Python package, upstream source or shared cache is changed by a
measurement. Prepare compatible native/JIT artifacts in private locations
before requesting a GPU lease.

The reviewed controller entry is `probe.py --run-plan <frozen-plan.json>`.
It requires `K31_TASK13_GPU_GRANT=<head>` and `K31_TASK13_PLAN_SHA=<sha256>`.
The external plan binds the command, exact environment, source/runtime files,
fresh output directory and 55-second deadline. It imports only the unchanged
`Owned` class from `/app/k31-agent-team-private/task-10/dependencies/owned-run.py`
(SHA256 `a102579f60f77f78faf47c3999cccc7d3b1b46dd7ed68f7ba40cf5fe583558ea`),
with the existing AST import contract and its pinned `owned-protocol.json`
(SHA256 `d3345964118cecd1d786824a38220258dad654af12a0040c662d8a0422158ca3`).
The caller supplies TERM3/KILL2 to that unchanged class. GPU-state primitives
come from `/share_nfs/whx/work/k31_perf_opt/.opt-team/harness/common.py`
(SHA256 `1479cc118a3d667cc19c570dae34f779c6c7fcac5df8bd061c6537a9fbddd3ab`).

The root-owned controller holds the existing GPU lock from admission through
verified release. It checks GPU_PAUSE and every card's idle memory, uses
CPU0-111 at normal priority, preserves combined stdout/stderr, and imposes a
55-second command deadline, TERM grace 3 seconds and KILL grace 2 seconds.
Only that invocation's process group is eligible for cleanup. Failed cleanup
retains the lock and evidence until root verifies release. Release requires no live owned PID/birth, no owned KFD entry and less than
2 GiB used at the selected BDF's sysfs allocation counter. Other GPU/process
state is recorded; it is never signalled by this controller. Bounded TERM/KILL
attempts precede an unbounded retained-lock release wait if verification fails.
The controller and child do not grant themselves a lease.

## Measurement boundaries

| Cell | Precision and working set | Timing and interpretation |
|---|---|---|
| Library compute | BF16 inputs/output, 8192 cubed, no bias; one A/B/output set | Existing GPU-event mean of 30 Python-issued calls after 5 warmups, repeated for 12 seconds. Windows starting in the first 3 seconds are excluded. Concurrent BDF-bound SYS clocks at 50 ms are retained. Report empirical torch.mm TFLOP/s with its sampled median clock. |
| A16W16 | BF16 8192 cubed, no bias/scales; original normal-test implementations | Original perftest warmup/iteration counts and profiler aggregate; graph=False, exactly one argument set. Full original numerical comparisons retained. |
| A4W4 | Packed MXFP4 inputs, E8M0 block32 scales and BF16 output at 2048x8192x8192; original preshuffled layouts | Original normal test including tuned and ASM controls; the unrelated split-K sweep is empty. Same fixed perftest policy as A16W16. |
| Streaming | Original FP16 read/copy at 2 and 8 GiB input | Original event timing, graph=False. Read rate counts input payload; copy counts logical read+write. Reduction stores are reported separately. |
| Cache transition | Original FP16 re-read sweep, 32 through 4096 MiB | Retains the 25 us eligibility cutoff. Report all rows, eligible indices and last sampled near-plateau size, or unresolved/no visible knee. This sweep cannot resolve 4 MiB L2. |

AITER profiler aggregates kernel time rather than callable wall time. Clocks
around its perftest include warmup/profiling and do not establish a sustained
A4W4 hardware roof. Input descriptors include shapes, strides, dtypes and
logical bytes; temporary allocations and library workspace remain outside
that descriptor sum. One argument set means no rotating-buffer HBM claim.
Keep each implementation's returned timing and configured kernel identity;
do not silently select the fastest observed implementation after the run.

Memory outputs are fully checked against Torch and for finiteness before
timing. AITER's reference and catastrophic checks are unchanged. Its
`checkAllclose` can log an ordinary mismatch without raising; the adapter
also rejects a returned ratio above that same call's threshold. Numerical
failure invalidates that command's rates. Library compute checks finite
output, while the separate AITER A16W16 control compares implementations.

Triton registers, memory-spill count, LDS and warp count are recorded when
exposed. SGPR usage and achieved occupancy are unavailable. Library and AITER
native resource mappings are unavailable unless a separate source-bound
binary receipt resolves them. Do not treat unavailable as zero or apply
gfx942 LDS/occupancy constants to gfx950. Theoretical TFLOP/s is null; dense
BF16 and A4W4 results use their own operation counts and no family percentage.

## Repetition and result

First prepare compatible artifacts, then run each command twice with the
same input seed, shape, source, cache policy and timing boundary. Every matched
rate/latency cell must agree within 2%, using absolute difference divided by
their mean. Otherwise diagnose the difference or report inconclusive. A warm
command taking a minute or longer does not meet the tight-loop criterion.

Then run five matched rounds in fixed roofline/A16W16/A4W4 order. Retain each
raw JSON and console log, the actual elapsed time, clock samples, numeric
checks and owned-release receipt. Report median, min, max and
`100 * (max - min) / median` for each same-boundary cell. Freeze the resulting
observed reproduction range before independent GPU reproduction; no range
is invented before those measurements exist. A cold build, failed command,
missing knee or absent clock/resource field is explicitly reported, never
converted into a successful ceiling or model/C12 benefit.
