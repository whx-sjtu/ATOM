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
