# Bounded C12 exposure

This spike exercises production serving, public profiler control and canonical
request aggregation. It measures the production base named by `protocol.json`;
its scripts close without merging. Read that file for the frozen workload,
precision choices, state policy and independent failure bounds.

Generate a plan without starting GPU work:

```sh
/share_nfs/whx/work/k31_perf_opt/.opt-team/harness/no_gpu.sh \
  /tmp/k31-agentteam-baseline.AvWp7t/venv/bin/python \
  spike/k31_c12_exposure/plan.py --out /absolute/fresh/run \
  --plan-out /absolute/frozen-plan.json
```

The orchestrator reviews the clean committed head, exact plan and complete
private runtime/dataset receipts before granting a lease. The grant supplies
`K31_TASK6_GPU_GRANT` (head) and `K31_TASK6_PLAN_SHA` (SHA256 of the plan file).
Run from the task checkout at normal priority on CPU0-111 with `/opt/venv/bin/python
spike/k31_c12_exposure/run.py --run-plan /absolute/frozen-plan.json`.
The driver checks pause/idle/lock, owns its child processes, preserves failures
without retry and retains the lock until owned processes and VRAM are released.

After release, run `analyze.py /absolute/run --build-bundles` through the CPU
wrapper and interpreter above, with `PYTHONPATH` set to the task checkout.
Keep its stdout, the run directory and canonical verification output. The
request-span denominator and full-response samples come from production
aggregation. A successful HTTP response with zero or missing output invalidates
the invocation. The diagnostic invocation stays separate from unprofiled
statistics; synthetic acceptance and the short-duration override remain visible.

Profiler qualification requires fresh ordered lifecycle evidence and parseable
exports from every rank within the conservative ten-second upper bound. Raw
kernel intervals and scheduled/running row labels are retained per rank. Host
scopes, overlapping families and rank sums do not establish GPU wall cost;
missing native MLA tail or embedded K2 communication boundaries remain unknown.
The report derives its reproduction range only from usable unprofiled samples.
None of these observations establishes official C12 performance or B300 headroom.
