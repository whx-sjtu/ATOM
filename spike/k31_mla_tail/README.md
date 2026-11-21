# Current-native MLA tail comparison

This spike compares the native BF16 V-up, graph mailbox/LSE merge and Kimi sigmoid gate with Inferact's fused equivalent. Both return gated BF16 rows before the external output projection. It constructs deterministic operator inputs; it does not load a model or qualify representative C12 behavior.

`adapter.py` calls the actual `MLAAttention._dcp_project_merge_out` with `decode=True`, an internal Identity projection, unscaled BF16 V-up weights and the existing Kimi gate helper. Eager native calls must take A2A fallback; a capture must increment the native mailbox launch count on every rank. The candidate receives the same inputs, replicated weights, per-row ownership and gate, with separate output storage. Per-token counts map to the donor mask through a unit-stride query-start array. No production function is replaced.

The candidate is the narrow Apache-2.0 package at Inferact/amd-kernels commit `78e27b0e75abb7e58a5b8d7bd6834300dd1ac90e`: `amd_kernels/attention/dcp8_lse_combine_vup_mla_h12_kv512`, its `common` helpers and `warm_up`, retaining their license files. The private archive is outside this branch. Timing follows that package's `benchmark_combine.py`: rotating inputs, captured calls, alternating arms and maximum rank time. Its vLLM reference is replaced by ATOM; its shared-output cosine section is replaced by separate-storage numerical checks. No donor bytes are changed.

The fixed contract is TP8/DCP8, latent512,96 real heads,12 output heads per rank,value128, BF16 values/weights/gate and FP32 base-e LSE. FP4/FP8 V-up weights or quantized gate outputs are outside it. Ordinary, empty, partially empty/padded and invalid-LSE cases run through repeated graph epochs before timing. Both outputs must be finite, global maximum absolute error divided by global maximum reference magnitude at most2^-7 and BF16 equality at least99.9%; empty rows must be exactly zero. The separate row-relative diagnostic records the donor denominator without replacing the inherited native gate. A correctness failure suppresses timing. Identical storage, NaNs and intentional numerical corruption are negative controls in the CPU check.

A communicator is created once per worker on its single stream before capture. The pinned donor explicitly allocates process-lifetime IPC memory without a free API. This harness does not invent a destructor: bounded workers terminate and the existing team supervisor records owned-process and GPU release. This does not qualify repeated in-process model destruction.

## CPU preparation

Use the team's CPU-only interpreter, `no_gpu.sh`, shared compile lock and private TMPDIR. `check` executes the current native dispatch body with controlled CPU helpers and the donor's CPU mask reference; it is not native GPU or graph evidence.

```bash
python spike/k31_mla_tail/run.py check
```

Commit the source, then freeze a fresh proposal. Source files, runtime/compiler bytes, dependency archives, environment, cache, shape matrix, command and finite bounds are hashed into the plan. The smoke stage uses token row44 first. The measurement stage covers8/16/20/36/40/44/48 with five alternating pairs; shape coverage is not a model-workload claim.

```bash
python spike/k31_mla_tail/run.py prepare --stage smoke --plan <fresh-plan> --out <fresh-output>
python spike/k31_mla_tail/run.py verify --plan <fresh-plan>
```

The spike/new-test/task gates are exempt for this measurement. Syntax, CPU boundary checks, frozen-plan checks and independent source/lifetime review precede a root GPU request. No plan authorizes GPUs.

## Root-authorized execution

Only after independent command qualification and an explicit root grant, set `K31_TASK14_GPU_GRANT` to the frozen head and `K31_TASK14_PLAN_SHA` to its independently checked plan hash, then use the plan's exact `command`. The runner reuses pinned team Owned PID/birth tracking and common spawn/stop primitives. It retains the shared lock until owned processes/KFD entries and every selected TP8 allocation are verified released. Failed observations quarantine the runner with its lock; cancellation records a stop and never escapes cleanup. The workload is bounded, while quarantine remains until verified release. Every rank is started through the pinned spawn helper with its own inherited lease FD and pre-import grant checks. Child TERM/INT are not blocked. A smoke pass is not the named timing result. Freeze a distinct measurement proposal only after the smoke result is understood.

Each rank records correctness, native graph witness, per-rank timings/skew, raw paired times, logical off-rank V/LSE payload bytes and verdict. Payload bytes exclude control flags and epoch synchronization traffic. The launcher retains compiler resource notes, logs and release evidence. Timing changes within2% or inconsistent across pairs are inconclusive; reproduction uses the observed maximum-rank distributions rather than historical numbers. First-call compilation is outside timing. Compiler-cache compatibility and resource collection must be checked on the frozen environment before execution is qualified.

Historical native-tail comparisons and P19 timings are not reused as current results. This experiment excludes g_proj scheduling changes, frontend/F1 correctness, external output projection, representative-model shape proof, full model quality and end-to-end C12 acceptance.
