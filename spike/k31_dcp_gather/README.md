# Fixed DCP cache-plane gather diagnostic

This experiment runs the matrix and bounds in `protocol.json`. It exercises
`atom.distributed.dcp_utils.get_dcp_group` and AITER's
`GroupCoordinator.all_gather(use_custom=False, dim=0)`. Identical TP/DCP rank
orders select the TP coordinator; workers record that identity and whether the
underlying TP/DCP groups share a process group.

`probe.py` generates rank-, plane- and repetition-specific contiguous bytes.
It enqueues data then scales on one current stream, without a synchronization
between them. After the pair's completion event, every worker compares every
output byte with a CPU concatenation in rank order. The parent grants the next
pair only after all workers verify the previous one. A mismatch, timeout or
unexpected route stops the run without a retry.

Run CPU validation from the repository root, with the approved CPU wrapper,
private HOME/TMPDIR and Python environment:

```sh
python spike/k31_dcp_gather/run.py --self-check
```

After committing, freeze a plan outside Git:

```sh
python spike/k31_dcp_gather/run.py --plan "$PLAN" --output "$RUN"
```

`$RUN` must be under the protocol's private `runs` directory and must not exist.
The plan pins the source, native-cache receipt, environment and child command.
The orchestrator must review it and grant that exact head and plan SHA before:

```sh
/opt/venv/bin/python spike/k31_dcp_gather/run.py --run "$PLAN"
```

The controller requires `K31_TASK7_GPU_GRANT=<head>` and
`K31_TASK7_PLAN_SHA=<plan SHA256>`, normal priority and CPU affinity within
0–111. It reuses the pinned resource helper for pause/lock/idle admission and
child creation. Communicator and pair deadlines are independently supervised;
the whole execution bound starts with child launch. PID/birth identities govern
TERM/KILL cleanup. Failed release or receipt writes retain the ownership lock.

Per-rank JSON and journals separate host enqueue breadcrumbs from device
completion. `controller.json` records the first failure, final rank results and
release evidence. PASS qualifies only fresh synthetic operands and this fixed
collective sequence. It does not clear prior stream work, physical pages,
operand ownership, a model run or C12 liveness. This spike closes without merge.
