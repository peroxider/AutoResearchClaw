"""Entry-point based pipeline extensions."""

from __future__ import annotations

import logging
from importlib import metadata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PRE_EXPERIMENT_HOOKS = "researchclaw.pre_experiment_hooks"
POST_RUN_HOOKS = "researchclaw.post_run_hooks"


def _entry_points(group: str) -> list[Any]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        selected = discovered.select(group=group)
    else:  # pragma: no cover - Python <3.10 compatibility
        selected = discovered.get(group, ())
    return sorted(selected, key=lambda entry_point: entry_point.name)


def run_extension_hooks(
    group: str,
    *,
    run_dir: Path,
    run_id: str,
) -> list[str]:
    """Run installed hooks in name order and raise on the first failure."""
    completed = []
    for entry_point in _entry_points(group):
        hook_name = f"{group}:{entry_point.name}"
        logger.info("[%s] Running extension hook %s", run_id, hook_name)
        try:
            hook = entry_point.load()
            result = hook(run_dir=run_dir, run_id=run_id)
        except Exception as exc:
            raise RuntimeError(f"extension hook {hook_name} raised: {exc}") from exc
        if result not in (None, 0):
            raise RuntimeError(f"extension hook {hook_name} returned {result!r}")
        completed.append(entry_point.name)
    return completed
