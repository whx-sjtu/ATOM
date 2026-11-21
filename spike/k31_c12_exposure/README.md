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

## Warmup observer (separate mode)

Freeze with `plan.py --mode warmup_observer --out <fresh-run> --plan-out <plan>`.
`run.py --run-plan <plan>` uses the existing owned supervisor and still requires
an exact reviewed head/plan grant. The plan binds all nested observer sources,
the native seams, configuration, and server/cache/client/metadata environments.
Only the server receives `worker_bootstrap` and recorder settings.
The plan freezes `AITER_ASM_DIR` with the trailing slash written by pinned
`aiter.jit.core`; worker environment and native-source comparisons remain exact. Native spawned
workers activate after device/group setup; nonworkers install passive import hooks.

The pre-load exporter/accessor probe and initial event records precede weight
loading. Two existing post-load all-gathers calibrate the selected-PG sequence
association from unique new FlightRecorder entries. The runtime layer/cache
catalog, initial event completion and selected-group history are sealed before
`_maybe_warmup`. No synthetic collective or event synchronization is inserted.

One AgentX invocation selects the first whole-batch singleton request at393216
keys and its401408-key successor (8192 new tokens each). The24 layers are3,7,...91,92.
Six current-stream events per call distinguish entry, the two gathers and their
intervening index-select, and final scatter. CPU-owned immutable host maps and
scalar tensor/view/stream descriptors do not prove consumed GPU values or lifetime.
Raw allocation/request generation and cross-run input correspondence stay unavailable.
Preparation respects native `ScheduledBatch` containers: new/context counts are
host NumPy arrays, cached counts and IDs are host lists. Integer snapshots reject
tensors before conversion. Flagged dummy/capture calls clear semantic state before
copying fields; the native empty idle batch remains a non-target snapshot.
CPU checks call native warmup/idle factories, Sequence/ScheduledBatch and
PublicationOwner, including mixed batches and both target lengths. Native capture
uses Context directly and bypasses prepare_model; its model/upload boundaries are
controlled in the CPU check, without executing a GPU graph.

Startup is bounded by900s; workload always ends bymin(readiness+420,launch+1320).
After a candidate starts, each rank's earliest incomplete frontier also has a30s
limit. The first completion/failure/phase/deadline stop is latched. The client
process group is killed before export grace; profiling overlap may already exist.
Evidence ends bymin(trigger+15,launch+1335), then normal TERM/KILL/release runs.
The parent does not wait on a worker's GIL, event query or exporter. A worker may
still have queued GPU work during grace. There is no retry or second arm.

Analyze with `analyze.py <run>`; observer mode rejects performance bundle creation.
The report preserves first unfinished communication and host-map bounds, using
per-layer tensor-view page capacity rather than shared-storage size. Complete
coverage remains inconclusive about the old failure; it never supplies an exposure
reproduction range or official result.

CPU checks (inside the declared container, under `no_gpu.sh` and the CPU lock):
`python -m pytest -q --confcutdir=spike/k31_c12_exposure spike/k31_c12_exposure/test_observe.py`.
For the native8-worker wiring probe, freeze a fresh observer plan and run
`test_observe.py` with `K31_CPU_ROUTE=1`, `K31_OBSERVER_PLAN=<plan>`, and the shim
at the front of PYTHONPATH. It uses real AsyncIOProcManager/spawn/AsyncIOProc and
the native ModelRunner constructor/setup call. Explicit fixtures replace external
kernels, device/PG/Event/exporter, model construction and IO queues; all8 distinct
workers must publish source-bound hook receipts while CUDA/distributed remain
uninitialized. This is wiring qualification only, never GPU readiness.
