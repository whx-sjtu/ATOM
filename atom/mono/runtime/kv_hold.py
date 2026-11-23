# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Hold resident-grid mono steps off while a KV connector's copies may run.

A mono kernel's CTAs, one a CU, spin on each other, and on gfx950 such a
kernel can stop for ever beside a copy kernel on another stream (ROCm/ATOM#3369;
the polling backoff only mitigates it). An LMCache MP transfer queues its
copies behind a producer fence recorded before the forward of the step that
dispatches it, so they run beside that forward and any later one until the
worker reports the transfer terminal.

The scheduler marks a step's connector metadata ``transfers_live`` while any
transfer it dispatched, this step's included, has not reported terminal. A
decode step so marked runs the production path: under FULL graphs the
``MONO_OFF_PHASE`` recording of its request bucket, eagerly the wrapper's
``hold_off``. The mark is scheduler metadata every TP rank receives alike, so
the ranks agree on every step (a rank that ran one mono step fewer would leave
its peers spinning in the next one's fence).
"""

import logging
import time
from collections import Counter
from contextlib import contextmanager

logger = logging.getLogger("atom")

MONO_OFF_PHASE = "mono_off"
HOLD_REASON = "LMCache transfer in flight"
SERVED = "served"
# routing counts are logged every this many decode steps
LOG_EVERY = 200


def transfers_live(batch) -> bool:
    """Whether the scheduler had a KV transfer dispatched and not reported
    terminal when it built ``batch``."""
    meta = getattr(batch, "connector_meta_output", None)
    return bool(getattr(meta, "transfers_live", False))


class MonoStepRouter:
    """Which recording a decode step replays, and the tally of why.

    ``model`` is the installed model; the router is inactive unless it is a
    mono wrapper (it has ``hold_off``) and ``enabled``."""

    def __init__(self, model, enabled: bool, log: bool = True) -> None:
        self.model = model if enabled and hasattr(model, "hold_off") else None
        # request bucket -> None when its recording runs mono, else why not
        self.bucket_reason: dict = {}
        self.counts: Counter = Counter()
        self._log = log
        self._steps = 0
        self._holding = False

    @property
    def active(self) -> bool:
        return self.model is not None

    def note_capture(self, key) -> bool:
        """Record what the forward just captured for ``key`` did; True when it
        ran mono, i.e. the bucket needs a ``MONO_OFF_PHASE`` recording too."""
        served = bool(getattr(self.model, "last_served", False))
        self.bucket_reason[key] = (
            None
            if served
            else (getattr(self.model, "last_decline", None) or "not served")
        )
        return served

    @contextmanager
    def held_off(self, reason: str = HOLD_REASON):
        """The wrapper declines every step inside, under ``reason``."""
        previous = self.model.hold_off
        self.model.hold_off = reason
        try:
            yield
        finally:
            self.model.hold_off = previous

    def graph_phase(self, batch, key, has_off: bool) -> str | None:
        """The phase a FULL-graph decode step of bucket ``key`` replays.
        ``has_off``: whether ``key`` has a ``MONO_OFF_PHASE`` recording."""
        reason = self.bucket_reason.get(key, "no recording")
        if reason is not None:
            self._count(f"declined: {reason}")
            return None
        if has_off and transfers_live(batch):
            self._hold(True)
            self._count(f"declined: {HOLD_REASON}")
            return MONO_OFF_PHASE
        self._hold(False)
        self._count(SERVED)
        return None

    def _hold(self, holding: bool) -> None:
        """Log each switch between mono and the held-off recording, with the
        wall time, so a step log can be told apart step by step."""
        if holding != self._holding:
            self._holding = holding
            if self._log:
                logger.info(
                    "Kimi-K3 mono %s at %.6f",
                    f"held off: {HOLD_REASON}" if holding else "resumed",
                    time.time(),
                )

    def eager_hold(self, batch, is_prefill: bool) -> str | None:
        """The ``hold_off`` an eager step runs under (prefill declines on its
        own and is not counted)."""
        if is_prefill:
            return None
        if transfers_live(batch):
            self._count(f"eager, declined: {HOLD_REASON}")
            return HOLD_REASON
        self._count("eager")
        return None

    def _count(self, what: str) -> None:
        self.counts[what] += 1
        self._steps += 1
        if self._log and self._steps % LOG_EVERY == 0:
            self.log()

    def log(self) -> None:
        logger.info(
            "Kimi-K3 mono step routing after %d decode steps: %s",
            self._steps,
            ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items())),
        )
