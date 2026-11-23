# LMCache producer-ready gate

Kimi-K3 mono beside LMCache MP (`kv_connector: lmcache_mp`) needs this gate in
the `lmcache server`. Without it a long serving run can wedge every rank.

```bash
export ATOM_LMCACHE_PRODUCER_READY=1
export PYTHONPATH=/path/to/ATOM/tools/lmcache_producer_ready:/path/to/ATOM:/path/to/aiter
lmcache server ...                                  # gated
python -m atom.entrypoints.openai_server ...        # serves mono beside lmcache_mp
```

Give the ATOM server the same two variables. There `ATOM_LMCACHE_PRODUCER_READY=1`
only states that the server beside it is gated; without it Kimi-K3 mono turns
itself off for `lmcache_mp` (`Kimi-K3 mono decode off: lmcache_mp without the
lmcache server's producer-ready gate`). The variable is read as `1` or not
`1`. The server logs `producer-ready gate installed on DefaultEventIPCBackend`
and `... on LMCacheDrivenTransferModule` once each, then its counts at the first
gated wait and at most once a minute after that (`transfers`: STORE / RETRIEVE
calls, `gated`: producer waits it held, `blocked`: of those, the ones whose
fence was not yet complete at the first query, `wait_s` / `max_wait_s`: host
time spent polling).

## The deadlock

A STORE or RETRIEVE makes the server's transfer stream wait on the producer
fence ATOM recorded on its compute stream (`import_event`, `wait_event`). On
ROCm a stream wait on an imported IPC event is not a CP barrier packet: CLR
launches `__amd_rocclr_streamOpsWait`, one work item that spins on a CU until
the event's signal is written (`IPCEvent::streamWait` ->
`KernelBlitManager::streamOpsWait`; `GPU_STREAMOPS_CP_WAIT` would need an HSA
signal behind the memory, which host-registered IPC shared memory does not
have).

ATOM records the fence where the transfer may start, behind work still queued
on its stream: for loads, before the step's forward, so behind the previous
step's graph. A mono step launches 256 CTAs of 512 threads, one per CU, that
wait on each other (`atom/mono/plan/execution.py`). If the waiter holds a CU
before all of a mono kernel's CTAs are placed, one CTA never is, the kernel
never finishes, the fence never signals, and the waiter never leaves. The
other ranks then wait on this one through the mono mailboxes.

## What the gate does

Inside `LMCacheDrivenTransferModule.store` / `.retrieve` it remembers the
producer handle the call was given. When `DefaultEventIPCBackend.wait_event`
is called with the event imported for that handle, it polls
`query_event(event)` (a host read) until the fence has completed, then makes
the original call. Any other wait is passed straight through, the block-ID
staging before the wait is unchanged, and the gate keeps no reference to the
event.

The server thread does not block for longer than before: the imported event is
dropped when the call returns, and its HIP destructor already blocks until the
event has completed. The transfer starts after the host has seen the fence
instead of the instant the GPU does, a poll interval (0.5 ms, 2 ms after the
first 100 ms) plus a launch later.

There is no deadline. A fence still incomplete after 10 s is logged, and again
every 60 s; raising would leave the client's request unanswered while the
unready event's destructor blocked anyway.

## Limits

* Only `DefaultEventIPCBackend` (ROCm and CUDA without LMCache's isolated IPC).
  Under isolated IPC (`TimelineSemaphoreEventIPCBackend`) nothing is gated,
  which shows as `gated` far below `transfers` in the counts.
* Only LMCache's `lmcache_driven` transfer mode.
* Copy kernels with no dependency on later ATOM work can still share CUs with a
  mono kernel; they only delay it (`tools/mono_mailbox_copy_repro.md`,
  `ATOM_MONO_POLL_SLEEP_UNITS`).
