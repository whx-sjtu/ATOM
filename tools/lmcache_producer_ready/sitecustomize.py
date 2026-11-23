# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Startup entry of the LMCache producer-ready gate (``producer_ready.py``);
the next ``sitecustomize`` on ``sys.path`` runs after it."""

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

if os.environ.get("ATOM_LMCACHE_PRODUCER_READY") == "1":
    from producer_ready import install_from_env

    install_from_env()

_here = Path(__file__).resolve().parent
# The harness's step-timing hook chains to this one; do not chain back to it.
_paths = [
    p
    for p in sys.path
    if Path(p or ".").resolve() != _here
    and not (
        Path(p or ".").name == "steptiming" and Path(p or ".").parent.name == "harness"
    )
]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _paths)
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
