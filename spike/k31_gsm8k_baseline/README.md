# Natural GSM8K baseline spike

This adapter leaves inference source unchanged. The protocol fixes the current
service, full GSM8K task and natural acceptance policy. Experimental scripts are
reviewed and reproduced, then closed without merging.

The CPU preparation receipt must bind source, runtime, model metadata, dataset,
prompts and each explicitly reused private cache file. File hashes do not assert
that the recorded target weight revision was freshly rehashed in full. The dataset
receipt states its content-hash algorithm separately from Arrow-file hashes.

Freeze a committed attempt with `run.py --freeze author-001`. The resulting plan
contains exact child commands, environment, paths and dependency hashes. Root grants
execution for that head and plan; CPU preparation is not GPU authorization. The
independent attempt follows the same startup and scoring policy in a fresh namespace.

Each full attempt qualifies startup, issues at most the declared smoke batch and
scores in the same service instance. No separate default startup-only model load is
performed. The built-in server request log preserves completed and partial request
evidence; no profiler or Python startup hook is installed. HTTP readiness does not
prove that the lazy CPU cache pool is fully resident.

Owned is extracted as an unchanged AST class from its hash-pinned source and uses
its pinned supporting protocol. The old controller and probe are never executed.
The enclosing finalizer retains the lock until owned processes are gone, all cards
are released and a timestamped receipt is durable. A failure to prove release puts
the attempt in quarantine, permitting only cleanup work.

The verifier requires the exact manifest-bound result and sample files, complete
document/filter coverage and matching task, document, prompt and target identities.
Wrong, empty and cap-limited answers remain evidence. Canonical A0 is the first
complete protocol-valid author attempt, including a low score. Independent paired
qualification never replaces it. Nonsignificance is not evidence of equivalence.

CPU preparation uses the absolute host guard and the private CPU environment.
`run.py --prepare author-001` archives the pinned sources, verifies individual cache
artifacts and creates the offline data/identity receipt. Run `pytest -q
spike/k31_gsm8k_baseline/test_contract.py` before freezing the committed head.
Set `K31_TASK10_CPU_ATTEMPT=independent-001` when checking the independent private
namespace. The full speculative configuration resolves only during the authorized runtime
check; CPU contracts consume EngineArgs and the actual lm_eval/API client.

Execution uses `K31_TASK10_GPU_GRANT=<head>` and `K31_TASK10_PLAN_SHA=<plan sha256>`
with `run.py --run <frozen plan>`, on normal-priority CPU0-111. These fields bind a
root grant; they do not grant a lease by themselves. Independent qualification
prepares `independent-001` and freezes that attempt at the same experimental head.

Use `verify.py --data <data receipt> --select <attempts.json>` for canonical A0,
and `--qualify <author controller.json> <independent controller.json>` for paired
qualification. Startup hardware, imports, full runtime configuration and graphs
remain checks of the separately authorized model execution. CPU fixtures establish
no GPU result or quality score.

The entry deadline covers plan hashes, cache inventory and admission. Cleanup writes
an immutable `release.json` before unlocking. Bulk artifact hashes and registry
finalization then run on CPU112-127 at nice19, without querying GPU state again.
An unfinished evidence commit blocks selection or a later author run. Once the
controller and registry are durable and the marker is removed, a marker-directory
sync error is surfaced with the completed outcome. Critical durability errors
still keep the pending gate. From the same clean experimental head, use the CPU guard with
`run.py --finalize <attempt output>` to recover its evidence within a bounded CPU
operation. Recovery reuses the original answers and release proof; an already
sealed attempt only retries the directory sync. It launches no model, does not
rewrite sealed answers and cannot replace a valid low score with a later result.
