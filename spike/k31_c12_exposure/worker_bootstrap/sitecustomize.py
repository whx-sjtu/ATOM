"""Passive worker import wiring; ordinary roles never have this directory."""

import importlib.util
import os
import sys
from pathlib import Path

if os.environ.get("K31_OBSERVER_PLAN"):
    path = Path(__file__).resolve().parents[1] / "observe.py"
    spec = importlib.util.spec_from_file_location("k31_observe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.install()
