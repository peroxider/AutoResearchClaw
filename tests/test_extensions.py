from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from researchclaw.pipeline import extensions


class _EntryPoint:
    def __init__(self, name: str, hook: Callable[..., object]) -> None:
        self.name = name
        self._hook = hook

    def load(self) -> Callable[..., object]:
        return self._hook


class _EntryPoints(list[_EntryPoint]):
    def select(self, *, group: str) -> _EntryPoints:
        del group
        return self


def test_extension_hooks_run_in_name_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def hook_a(*, run_dir: Path, run_id: str) -> int:
        calls.append(("a", run_dir, run_id))
        return 0

    def hook_b(*, run_dir: Path, run_id: str) -> None:
        calls.append(("b", run_dir, run_id))

    entry_points = _EntryPoints([
        _EntryPoint("20_b", hook_b),
        _EntryPoint("10_a", hook_a),
    ])
    monkeypatch.setattr(extensions.metadata, "entry_points", lambda: entry_points)

    completed = extensions.run_extension_hooks(
        extensions.POST_RUN_HOOKS,
        run_dir=tmp_path,
        run_id="run-1",
    )

    assert completed == ["10_a", "20_b"]
    assert calls == [("a", tmp_path, "run-1"), ("b", tmp_path, "run-1")]


def test_extension_hook_nonzero_result_blocks_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry_points = _EntryPoints([_EntryPoint("bad", lambda **_kwargs: 2)])
    monkeypatch.setattr(extensions.metadata, "entry_points", lambda: entry_points)

    with pytest.raises(RuntimeError, match="returned 2"):
        extensions.run_extension_hooks(
            extensions.POST_RUN_HOOKS,
            run_dir=tmp_path,
            run_id="run-2",
        )
