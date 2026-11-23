# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Admit LMCache's GPU wait on an ATOM producer fence only once the fence has
completed (``ATOM_LMCACHE_PRODUCER_READY=1``, in the ``lmcache server``).

LMCache MP's STORE and RETRIEVE make the server's transfer stream wait on the
interprocess event ATOM recorded on its compute stream (``import_event``, then
``wait_event`` -> ``Event.wait``). On ROCm a stream wait on an imported IPC
event is a one-work-item kernel (``__amd_rocclr_streamOpsWait``) that spins on a
CU until the event signals. ATOM records the event behind work that may still
be queued, and a mono step's kernels need every CU at once
(``atom.mono.plan.execution.BLOCKS``): a CU held by the spinning waiter keeps
one of their CTAs from ever being placed, so the step never finishes and the
event the waiter spins on never signals.

The gate polls ``Event.query`` (a host read of the event's shared signal) until
the imported producer event has completed, then issues the original GPU wait,
which no longer spins. Only the event imported for the handle of the STORE or
RETRIEVE in progress is gated; every other wait is untouched. The host does not
block for longer than before: the imported event's destructor, at the end of
the same call, already blocks until the event completes.

There is no deadline. Raising would leave the client's request unanswered while
the unready event's destructor blocks anyway, so a fence that does not complete
is reported (`WARN_AFTER_S`, then every `WARN_EVERY_S`) and waited for.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import inspect
import logging
import os
import sys
import threading
import time

ENABLED_ENV = "ATOM_LMCACHE_PRODUCER_READY"
EVENT_IPC = "lmcache.v1.platform.base.event_ipc"
TRANSFER = "lmcache.v1.multiprocess.modules.lmcache_driven_transfer"
_TARGETS = frozenset((EVENT_IPC, TRANSFER))

# A producer fence trails the step in flight: a decode step is tens of ms, a
# prefill step up to about a second.
POLL_S = 0.0005
SLOW_POLL_S = 0.002
SLOW_AFTER_S = 0.1
WARN_AFTER_S = 10.0
WARN_EVERY_S = 60.0
STATS_EVERY_S = 60.0

_SCOPE = threading.local()
_STATS_LOCK = threading.Lock()
_STATS = {"transfers": 0, "gated": 0, "blocked": 0, "wait_s": 0.0, "max_wait_s": 0.0}
_stats_logged_at = None
_log = logging.getLogger("lmcache.atom_producer_ready")


def _logger():
    """LMCache's loggers do not propagate, so take one the way it builds its own."""
    global _log
    if not getattr(_logger, "done", False):
        _logger.done = True
        try:
            from lmcache.logging import init_logger

            _log = init_logger("lmcache.atom_producer_ready")
        except Exception:  # noqa: BLE001 - keep the stdlib logger
            if not _log.handlers:
                _log.addHandler(logging.StreamHandler())
                _log.setLevel(logging.INFO)
    return _log


def poll_ready(query, *, describe=lambda: "", clock=time.monotonic, pause=time.sleep):
    """Poll ``query()`` until it is true; returns ``(polls, seconds)``.

    Exceptions from ``query`` propagate unchanged."""
    start = clock()
    warn_at = WARN_AFTER_S
    polls = 0
    while True:
        polls += 1
        if query():
            return polls, clock() - start
        waited = clock() - start
        if waited >= warn_at:
            _logger().warning(
                "producer-ready gate: producer fence not complete after %.0f s%s; "
                "GPU wait not admitted yet",
                waited,
                describe(),
            )
            warn_at += WARN_EVERY_S
        pause(POLL_S if waited < SLOW_AFTER_S else SLOW_POLL_S)


def _frame():
    stack = getattr(_SCOPE, "stack", None)
    return stack[-1] if stack else None


def _count(clock=time.monotonic, **delta):
    """Add to the counts; on a gated wait, log them if none were logged in
    the last `STATS_EVERY_S` (so at the first one)."""
    global _stats_logged_at
    with _STATS_LOCK:
        for key, value in delta.items():
            _STATS[key] += value
        if "wait_s" in delta:
            _STATS["max_wait_s"] = max(_STATS["max_wait_s"], delta["wait_s"])
        if not delta.get("gated"):
            return
        now = clock()
        if _stats_logged_at is not None and now - _stats_logged_at < STATS_EVERY_S:
            return
        _stats_logged_at = now
        snapshot = dict(_STATS)
    _logger().info("producer-ready gate: %s", snapshot)


def stats() -> dict:
    with _STATS_LOCK:
        return dict(_STATS)


def _wrap_transfer(original):
    """STORE / RETRIEVE: note the producer handle the call was given."""
    signature = inspect.signature(original)
    if "event_ipc_handle" not in signature.parameters:
        raise TypeError(f"{original.__qualname__} takes no event_ipc_handle")

    @functools.wraps(original)
    def transfer(*args, **kwargs):
        handle = signature.bind_partial(*args, **kwargs).arguments.get(
            "event_ipc_handle"
        )
        stack = getattr(_SCOPE, "stack", None)
        if stack is None:
            stack = _SCOPE.stack = []
        stack.append({"handle": handle, "event": None})
        _count(transfers=1)
        try:
            return original(*args, **kwargs)
        finally:
            stack.pop()

    return transfer


def _wrap_import(original):
    @functools.wraps(original)
    def import_event(self, handle, device):
        event = original(self, handle, device)
        frame = _frame()
        if frame is not None and frame["event"] is None and handle == frame["handle"]:
            frame["event"] = event
        return event

    return import_event


def _wrap_wait(original):
    @functools.wraps(original)
    def wait_event(self, event, stream):
        frame = _frame()
        if frame is None or frame["event"] is not event:
            return original(self, event, stream)
        frame["event"] = None
        polls, seconds = poll_ready(
            lambda: self.query_event(event),
            describe=lambda: f" (pid {os.getpid()}, thread {threading.get_ident()})",
        )
        _count(gated=1, blocked=int(polls > 1), wait_s=seconds)
        return original(self, event, stream)

    return wait_event


def _patch(module):
    if getattr(module, "_atom_producer_ready_patched", False):
        return
    if module.__name__ == EVENT_IPC:
        cls = module.DefaultEventIPCBackend
        cls.import_event = _wrap_import(cls.import_event)
        cls.wait_event = _wrap_wait(cls.wait_event)
    else:
        cls = module.LMCacheDrivenTransferModule
        cls.store = _wrap_transfer(cls.store)
        cls.retrieve = _wrap_transfer(cls.retrieve)
    module._atom_producer_ready_patched = True
    _logger().info(
        "producer-ready gate installed on %s (pid %d)", cls.__qualname__, os.getpid()
    )


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install_from_env() -> bool:
    """Patch the two LMCache modules when (or if) they are imported; a no-op
    unless ``ATOM_LMCACHE_PRODUCER_READY=1``. Only the server runs STORE and
    RETRIEVE, so in any other process the wrappers pass straight through."""
    if os.environ.get(ENABLED_ENV) != "1":
        return False
    if getattr(sys, "_atom_producer_ready_installed", False):
        return True
    for name in _TARGETS:
        if name in sys.modules:
            _patch(sys.modules[name])
    sys.meta_path.insert(0, _Finder())
    sys._atom_producer_ready_installed = True
    return True
