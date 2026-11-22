# Current whole-K2 readiness floor

This spike compares one actual current K2 invocation with an optimistic X-ready
floor. It changes no production source. The result is not production correctness,
quality, a serving gain or an absolute hardware ceiling.

The treatment is current Mono enabled, TP8/DCP8, K3/AL2.88, native precision and
the fixed state/cache controls in the bound current recipe. Historical #519 used
Mono disabled and a different recipe; its metrics are not this denominator.
The capture stimulus is one fixed real request, not a concurrency or MTP search.

## Capture and state

The declared candidate is non-block-writing KDA layer17, request bucket(1,4),
four actual rows, one16-row group and two read blocks. Startup dummy calls do not
qualify. One ordinary live Mono replay must be witnessed first; only the next
matching real invocation uses the named one-shot capture graph. An unexposed
path or a different key fails rather than changing K or manufacturing S5.

The existing storage/range/hash/copy machinery retains exact per-rank core,
prefix-before, read blocks with aliases/strides, generated X, prefix-after,
materialized weight shards and reference output/ordered routes/weights. All
eight files must rejoin one epoch/invocation and pass current ABI validation.
Test fixtures and incomplete captures are never timing inputs.

The floor factory clones the current build's globals into a private namespace
and private build cache. Only o_proj/AttnRes producer bindings are compile-time
no-ops. Downstream routing, experts, shared work, tail, XRDY polls and barriers
remain native. Its symbol/cache provenance includes the adapter/stub bytes.
CPU qualification compares original-binding generated source/IR/text with the
direct baseline, then checks the floor and failure paths before baseline again.

## Replay and interpretation

Every warmup, check and timing iteration restores consumed state. begin_step
clears/fences mailboxes before saved X and tagged XRDY are published. A uses
prefix-before; B uses the corresponding prefix-after. Graph-visible addresses
and immutable weights remain fixed. A/A and A/B output, prefix, X, ordered
route IDs/weights and U must agree with capture before timing.

Ordinary events enclose one K2 graph. Restoration/seeding are outside the
envelope. Existing D/2D, clock validity, global submission and whole-pair retry
guards are retained. Each delay condition has five alternating AB/BA pairs
and A/A controls. Pair max(A ranks) minus max(B ranks) before summarizing.
Record rejected pairs and resource/occupancy changes; subtract no setup or
guessed graph overhead. A delta within2% is inconclusive. Reproduction ranges
come from author spread/A/A noise and are frozen before independent measurement.

The floor includes removed producer traffic/publication and compiler-resource
effects. It is not pure wait time or an implementable free input source.
Tile shape, placement, grouping, prefetch and layout remain fixed; wider shapes,
different inputs and whole-model benefit remain untested.

## Ownership and entry points

All GPU phases need an exact root head/plan lease and whole-node admission.
The reused ownership body signals only PID/birth-proven owned processes.
Release requires their exit, exactly eight valid idle card allocations and an
independently valid empty global KFD list. KFD is sampled before SMI; no
container-PID intersection or shared-helper relaxation is used. Unknown release
retains the lock. Ownership checks use CPU fakes; compile qualification runs
under the device-open guard.

From the repository root, use the approved no_gpu.sh wrapper for source checks
and the pinned guarded compiler environment for compile-only preparation:

    python spike/k31_k2_front/replay.py --source-check --out NEW_CPU_OUTPUT
    python spike/k31_k2_front/replay.py --compile-only --out NEW_COMPILE_OUTPUT
    python spike/k31_k2_front/replay.py --write-plan capture --out NEW_PRIVATE_RUN --plan NEW_PLAN
    python spike/k31_k2_front/replay.py --cpu-check --snapshot COMPLETE_CAPTURE --out NEW_READY
    python spike/k31_k2_front/replay.py --write-plan timing --snapshot COMPLETE_CAPTURE --ready READY_JSON --out NEW_PRIVATE_RUN --plan NEW_PLAN

Plan creation does not authorize execution. Root alone invokes --run-plan after
independent qualification. Source/recipe/helper hashes and private cache/input
identities accompany the plan. Retain failed artifacts; no automatic job retry.

Before worker imports, the plan verifies current source/helper hashes, installed
runtime distribution bytes, AITER sources and preexisting private cache artifacts.
Model identity binds metadata bytes plus shard filesystem/header identities;
selected materialized weight bytes are hashed by the real capture. Full engine
configuration and startup can require a device and remain lease-bound checks.

## Prospective clock-boundary diagnostic

The separate `clock_boundary` plan runs no model, snapshot or K2 path. It keeps
all eight visible device mappings, original Timer event/delay priming, one graph
warmup, the same 0.5 ms body, 1 ms/2 ms guards, and exactly two trials per delay.
An untraced cohort precedes a traced cohort, at most 64 measured rows in total.
There is no `collect_pair`, retry, replacement sample or backend/counter search.
Each worker records its HIP BDF before Timer priming; untraced identities must
form the fixed eight-device bijection, and traced ordinals must match it.

The original event gate remains `E + 5 >= B` and
`E <= B + max(50, 0.10 * B)` in microseconds. A finite mismatch is recorded as
a failed observation while the fixed set may finish. Content, counter, order,
submission, trace, ownership or deadline failures stop subsequent observations
and cohorts. The absolute 180 s bound starts before the first diagnostic child
spawn and covers both cohorts, workers, profiler exit/flush and in-run validation.
Late exit or late validation fails even if the child returned zero. Owned cleanup
uses bounded TERM/KILL and retains the lock when release is uncertain.

The fixed rocprofv3/SDK backend traces HIP and kernel records to JSON. Strict
invalid-timestamp rejection is enabled and frequency scaling is disabled. Each
leased worker requires a successful HSA timestamp-frequency query of 1 GHz before
interpreting unscaled dispatch ticks as nanoseconds. Resolved loader mappings,
raw graph/stream handles, native TID, PID and actual BDF are recorded. The required
lineage is five graph calls (warmup ordinal 0, measured ordinals 1–4), five
priming/guard dispatches and six module calls, including one capture-definition
call without a dispatch. Exact internal correlation must close each required
call; no nearest-time, duration, ancestor or alias fallback is allowed.

All visible SDK loss, repair, warning and error diagnostics fail, including
unrelated domains. The sole log exception is the SDK's exact expected-file
`Opened result file` success message. Complete normal process exit, complete
JSON and independent required counts are required; `metadata.fini_time` is not
flush completion. Unmodelled unrelated records are retained, without a claim
that every unrelated exported record was lossless.

Run CPU fixture checks and plan creation through the approved device-open guard:

    python spike/k31_k2_front/replay.py --clock-source-check
    python spike/k31_k2_front/replay.py --write-plan clock_boundary --out NEW_PRIVATE_RUN --plan NEW_PLAN

Plan creation does not grant execution. After independent Source qualification,
root alone may issue the exact head/plan lease for one set. Only after verified
owned release does offline analysis pair measured rows with their exact dispatch:

    python spike/k31_k2_front/replay.py --analyze-clock --plan FROZEN_PLAN --out NEW_ANALYSIS

B spans first `memrealtime` through the last loop read, excluding the two stores
and return. K is the SDK/HSA dispatch processing/completion duration; E is the
unchanged ordinary event envelope. Report raw durations and K−B/E−K, without
absolute counter/SDK-clock alignment, overhead subtraction or physical-cause
attribution. Non-nesting, trace-only behavior, unmatched records and nonrecurrence
remain unresolved. Seal, review and stop after the one set, including success;
there is no automatic method adoption or K2 continuation. Earlier failed clock
observations and the zero-K2 result are unchanged.
