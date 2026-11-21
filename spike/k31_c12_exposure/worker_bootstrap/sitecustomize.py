"""Passive worker import wiring; ordinary roles never have this directory."""

import importlib.util
import os
import sys
from pathlib import Path

selected = [
    (key, filename, name)
    for key, filename, name in (
        ("K31_OBSERVER_PLAN", "observe.py", "k31_observe"),
        ("K31_SAVE_OBSERVER_PLAN", "save_observe.py", "k31_save_observe"),
    )
    if os.environ.get(key)
]
if len(selected) > 1:
    raise RuntimeError("observer modes cannot share a service")
if selected:
    key, filename, name = selected[0]
    path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.install()
