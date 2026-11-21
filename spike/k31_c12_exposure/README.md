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

The prospective coverage contract is issue6 ruling7057415189. It does not
reinterpret the sealed0c83 deadline result. L is the original supervisor launch
anchor, R sealed readiness, C the client-spawn call (return is separately recorded),
and W first controller observation of an actual warmup-phase start. Producer log
time and discovery delay are separate, with their clock assumption retained.
Startup ends by L+900. Setup must reach W by min(C+150,R+420,L+1320); all eight
workers must then join the same recorded393216 first-helper entry by
min(W+240,R+420,L+1320). Headers/maps and unrecorded events cannot retire this cap.
Only a matching quorum switches coverage to L+1320, without resetting L. T records
the earliest entry and the parent quorum-observation time is retained separately.
Recorded streams are checked within each rank; raw handles are not compared across
processes as if they were global identities.

The per-rank earliest-incomplete30s rule and first-stop latch remain unchanged.
The actual profiling-phase transition stops the client even if traffic preceded
its log. The existing hard evidence alarm ends collection by
min(trigger+15,L+1335), then the same TERM15/KILL15/retained-lock release runs.
There is no observer-mode900s client timer. Inner AgentX and native600s watchdog
settings are unchanged. Signal/failure paths preserve actual process return/censor
and phase evidence in a bounded client receipt and final cleanup status. Available
export/progress/terminal fields are kept separate; missing cancellation counts
remain unknown. Missing ledger data does not delay cleanup or imply a second arm.

Observer costs are fixed-size aggregate categories with started/finished/failed,
in-flight/max-in-flight, inclusive wall/thread-CPU sums/maxima, completed payload bytes and
query outcomes (byte-null categories have no measured byte quantity; these are
not GPU traffic counts). They cover host snapshots/descriptors, marker calls, initial and
pending queries, the query sweep, JSON/file/publication work and each explicit FR
call. Original native calls are outside these scopes. A successful startup still
has six explicit FR calls but one startup file; the target has no FR calls before
completion/trigger and one final export. The full pending sweep, work+0.1s sleep
and full-state publication policy are unchanged.

The categories overlap/nest and wall time may include waiting/preemption; they
must not be summed or subtracted into native/GPU durations. Snapshots precede their
own encode/write completion, so the current publication appears in flight and its
child costs need a later snapshot. Unfinished calls, missing snapshots and partial
failed-write bytes remain unavailable. Timing/aggregation overhead and some setup
bookkeeping are not separately measured; host accounting cannot prove zero device
perturbation. No per-query file writes, localization marker or on/off arm is added.

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

## Native save completion observation

Freeze a separate plan with --mode save_completion_observer. It uses the existing
execute/--run-plan ownership lifecycle and one first-client-equivalent invocation.
Only serving processes receive K31_SAVE_OBSERVER_PLAN and the passive shim.
The indexer observer, GPU markers, FlightRecorder calls and profiler controls are
inactive. Client, LMCache and metadata roles retain ordinary environments.

The prospective selector is logical request 9 and its first eight distinct actual
save generations. Generation 111 or 340 and cross-run input correspondence are
never assumed. One save in flight does not bound the request's total generations;
cap exhaustion stays inconclusive without a sliding window or budget reset.
The bfd2bd10 req130 experiment retains its original FAILED/inconclusive result. Native
submission, future queries/results, existing aggregation and settlement calls keep
their original arguments, results, exceptions and order. In particular, the
installed result(timeout=0) path keeps its existing event synchronization; the
observer adds none. The raw/event query scope excludes adapter housekeeping for
other outstanding futures. Per-poll emission rows identify newly added
completions for that poll's operation.

Source-bound constructor receipts identify EngineCore/manager and eight workers
before traffic. Diagnostic ordinals count existing broadcasts/executions under
the native sole-outstanding/FIFO contract. Consumed and newly started snapshots
remain distinct. Queue reads are observed where native code consumes them; no
payload or wire identifier is added.

Each role writes one capped JSON snapshot containing bounded transition records,
coalesced scalar first/last/count summaries and current/previous aggregation state.
Repeated false queries do not append trace rows or perform file writes. Publication
uses existing aggregation/startup/failure boundaries. Worker records are capped at
256/128KiB each; the manager at 512/256KiB, for 1.25MiB of retained role payload.
In-place writes keep no duplicate temporary payload; interrupted or failed writes
are unavailable. Overflow is explicit and never adds a workload stop or retry.
A failed stop-control write preserves the original terminal state/error and is
recorded separately in status.json. Both the status summary and offline reducer
mark the join unavailable because the shortened worker cutoff was not proved.
Existing fatal native exceptions publish one scalar global failure header
independently of the selector or generation/record admission, then re-raise the
original exception. A byte-capped snapshot retains its source/role identity and
fatal header within the existing cap while detailed evidence is unavailable.
The current controller consumes that notification even if the frontend remains
alive. Recorder write failure stays unavailable and preserves the native exception;
it cannot establish successful notification delivery.
Recorder-owned cost counters exclude native calls, overlap and omit their own
publication tail; cumulative I/O bytes are separate from the retained payload cap.

The supervisor preserves metadata60s, existing readiness900s and the client120s
profiling phase with ordinary warmup/seed/grace. Its independent bounds are
min(C+900,L+1800) for the client and L+1800 for the workload. No readiness,
selection or generation resets them. Normal invocation end, existing failure,
signal or deadline latches once. Collection ends at min(stop+15,L+1815), then
the existing TERM15/KILL15 and retained-lock release checks apply. Native
worker30s plus scheduler10s remain unchanged. Partial records, native settlement
and client cancellations remain distinct.

After owned release, run save_observe.py <run> --report-out <new-evidence-file>
through the CPU guard. This narrow reducer reports keyed chains and missing joins;
it never constructs an exposure/performance bundle. Missing activation, overflow,
request absence, semantic mismatch or ordinal/generation disagreement is
inconclusive. Source/plan review and a fresh root lease remain mandatory.
