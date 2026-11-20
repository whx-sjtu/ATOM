---
repo: whx-sjtu/ATOM
integration_branch: k31-perf-opt
remote: origin
gate_task: python tools/agent_team/gate.py
design_entry: docs/architecture_guide.md
never_touch: ROCm/ATOM, ROCm/aiter
new_tests_dir: tests
publish_language: English
conversation_language: Chinese
max_tasks: 8
gate_jobs: 1
split_loc: 1000
task_label: k31-performance
import_check: python -c "from pathlib import Path; import atom; p=Path(atom.__file__).resolve(); print(p); assert p.is_relative_to(Path.cwd().resolve() / 'atom')"
---

This workstream implements the owner-approved C12 milestone recorded in whx-sjtu/ATOM#1, under wuhuikx/LLM-Booster#437. GitHub issues, PRs and their handoffs are the task record. Keep raw experimental artifacts and scratch outside tracked source, with hashes and locations in the corresponding GitHub record.

Use GPT-6.1 Sol with xhigh reasoning for every subagent. Eight is the global active-agent cap across both forks and all roles; never interpret max_tasks as permission to exceed that cap. Prioritize review and execution readiness over starting another author when slots are full. The author, reviewer and orchestrator remain distinct.

Land only on k31-perf-opt. The ATOM and AITER upstream repositories are read-only. A necessary AITER change is developed in whx-sjtu/aiter with its own task/PR and exact dependency commit recorded by the consuming ATOM task. Cross-repository dependencies must be explicit; hold landing until their required reviewed state is satisfied.

The gate runs the existing non-GPU suite, Black check and the approved baseline-aware Ruff check. Existing findings are recorded, not called clean; new findings, tool errors and unapproved allowance growth fail. Do not turn unrelated lint cleanup into this milestone's scope.

Measurement tooling that leaves default inference unchanged may land after the CPU gate and independent review. Inference or numerical optimizations additionally require the GPU correctness, performance and established quality evidence named in their issues before APPROVE and landing. A compiled binary, skipped CI job, microbenchmark, older-head result or unobserved code path does not establish model benefit.

Retain the frozen workload, precision and quality agreements in the umbrella. Natural-acceptance research and the official golden-acceptance protocol are different experiments. Preserve source/runtime identities, timing boundaries, actual batch and committed-token counts, failure evidence and cleanup results. Do not relabel historical research results as new-main evidence.

The owner delegated subsequent recommended decisions while away. Use automatic rulings for technical decisions within the approved agreement rather than waiting for routine confirmation. Do not silently change targets, precision limits, repository boundaries or resource ownership. Report only key progress and material blockers to the owner.

GPU runs are scheduled by the orchestrator under the local environment's ownership rules. Complete compatible-cache preparation before consuming the GPU lease. Mechanism validation and official acceptance follow their agreed tier, and expensive official runs occur only at meaningful milestones.
