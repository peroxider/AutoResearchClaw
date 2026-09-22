"""End-to-end tests for the 5-layer paper-fabrication defenses.

These tests cover the in-process backstops installed after the paper_2
incident, where a stub ``main.py`` produced zero real metrics, Stage 13
synthesised 800+ fabricated metric values, and Stage 17 quoted them
verbatim into ``paper_draft.md``.

The defenses are:

  Layer 1 — ``_code_generation._validate_generated_main_py`` rejects
            config-only stubs before Stage 10 returns DONE.
  Layer 2 — ``_execution._execute_experiment_run`` returns FAILED when
            the run consumed <2% of its time budget and produced <5
            metric keys (the paper_2 pattern: 0.3 s on a 600 s budget).
  Layer 3 — ``runner._drop_untraceable_summary_values`` scrubs
            ``experiment_summary_best.json`` to only the values that
            appear in any ``stage-*/runs/*.json`` payload.
  Layer 4 — ``_paper_writing._audit_draft_for_fabrication`` audits the
            just-written ``paper_draft.md`` and rejects the stage when
            >10% of metric-shaped numbers cannot be traced to
            VerifiedRegistry.
  Layer 5 — ``config.validate_config`` rejects ``experiment.mode ==
            'simulated'`` and ``research.quality_threshold < 3.0``,
            warns on short empirical-topic budgets.

These are unit-level: each test invokes the defensive function with
the minimum required run_dir / config and checks the verdict, without
needing to run the full 23-stage pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Layer 1 — Stage 10 stub detector (in-process backstop in _code_generation)
# ---------------------------------------------------------------------------

class TestStage10StubDetector:
    """``_validate_generated_main_py`` must reject config-only stubs."""

    def _validate(self, src: str) -> tuple[bool, str]:
        from researchclaw.pipeline.stage_impls._code_generation import (
            _validate_generated_main_py,
        )
        return _validate_generated_main_py(src)

    def test_config_only_stub_is_rejected(self) -> None:
        # paper_2's actual Stage 10 emit (a Config class, nothing else)
        ok, detail = self._validate("class Config:\n    seed = 42\n")
        assert not ok
        assert "def run" in detail

    def test_empty_main_is_rejected(self) -> None:
        ok, detail = self._validate("")
        assert not ok
        assert "empty" in detail.lower()

    def test_short_body_with_imports_is_rejected(self) -> None:
        # Imports numpy but has only a 2-line `run()` body — under the
        # 30-statement module threshold, the synthetic-number generator
        # would still get away with it.
        src = (
            "import numpy as np\n"
            "def run():\n"
            "    pass\n"
            "if __name__ == '__main__':\n"
            "    run()\n"
        )
        ok, detail = self._validate(src)
        assert not ok
        assert "minimum" in detail or "statements" in detail

    def test_no_numerical_imports_is_rejected(self) -> None:
        # Long body, but only stdlib imports → still a stub.
        body = (
            "import json\n"
            "import sys\n"
            "\n"
            "def helper():\n"
            "    a = 1\n"
            "    b = 2\n"
            "    c = 3\n"
            "    d = 4\n"
            "    e = 5\n"
            "    f = 6\n"
            "    g = 7\n"
            "    h = 8\n"
            "    return a + b + c + d + e + f + g + h\n"
            "\n"
            "def setup():\n"
            "    cfg = {}\n"
            "    cfg['seed'] = 42\n"
            "    cfg['limit'] = 100\n"
            "    cfg['offset'] = 0\n"
            "    cfg['scale'] = 1.0\n"
            "    cfg['bias'] = 0.0\n"
            "    cfg['noise'] = 0.01\n"
            "    cfg['folds'] = 5\n"
            "    cfg['workers'] = 4\n"
            "    cfg['tag'] = 'test'\n"
            "    return cfg\n"
            "\n"
            "def emit(results):\n"
            "    keys = list(results.keys())\n"
            "    keys.sort()\n"
            "    top = keys[:5]\n"
            "    for k in top:\n"
            "        v = results[k]\n"
            "        print('METRIC: ' + str(k) + '=' + str(v))\n"
            "    return top\n"
            "\n"
            "def run():\n"
            "    cfg = setup()\n"
            "    results = {}\n"
            "    for i in range(50):\n"
            "        results[i] = helper() + i + cfg['offset']\n"
            "        for j in range(20):\n"
            "            results[i] += j\n"
            "    emit(results)\n"
        )
        ok, detail = self._validate(body)
        assert not ok
        assert "numerical" in detail.lower() or "library" in detail.lower()

    def test_valid_main_function_entry_passes(self) -> None:
        # Regression for paper_1: the real driver uses ``def main():`` (with
        # ``sys.exit(main())`` in the ``if __name__`` block), not the
        # conventional ``def run():``. The validator must accept either
        # entry-point name to avoid blocking legitimate drivers.
        src = (
            "import json\n"
            "import sys\n"
            "import time\n"
            "\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "def _load():\n"
            "    df = pd.read_csv('/data/data.csv')\n"
            "    X = df[['a', 'b']].to_numpy(dtype=np.float64)\n"
            "    y = df['y'].to_numpy(dtype=np.int64)\n"
            "    return X, y\n"
            "\n"
            "def _fit(X, y):\n"
            "    scaler = StandardScaler().fit(X)\n"
            "    clf = LogisticRegression(max_iter=400).fit(scaler.transform(X), y)\n"
            "    return scaler, clf\n"
            "\n"
            "def _eval(scaler, clf, X, y):\n"
            "    p = clf.predict_proba(scaler.transform(X))[:, 1]\n"
            "    return {'auc_roc': float(np.mean(p[y == 1])), 'n': int(len(y))}\n"
            "\n"
            "def _emit(prefix, metrics):\n"
            "    for k, v in metrics.items():\n"
            "        print(f'{prefix}_{k}: {v}')\n"
            "\n"
            "def _config():\n"
            "    cfg = {}\n"
            "    cfg['seed'] = 42\n"
            "    cfg['split'] = 0.2\n"
            "    cfg['max_iter'] = 400\n"
            "    cfg['tag'] = 'paper_1'\n"
            "    return cfg\n"
            "\n"
            "def main() -> int:\n"
            "    start = time.time()\n"
            "    X, y = _load()\n"
            "    scaler, clf = _fit(X, y)\n"
            "    metrics = _eval(scaler, clf, X, y)\n"
            "    _emit('train', metrics)\n"
            "    _emit('val', metrics)\n"
            "    print(f'elapsed: {time.time() - start:.2f}s')\n"
            "    return 0\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    sys.exit(main())\n"
        )
        ok, detail = self._validate(src)
        assert ok, detail
        assert "def main()" in detail

    def test_valid_driver_passes(self) -> None:
        # A realistic 30+ statement driver that imports sklearn + numpy.
        src = (
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "def _load():\n"
            "    df = pd.read_csv('/data/data.csv')\n"
            "    X = df[['a', 'b', 'c']].to_numpy(dtype=np.float64)\n"
            "    y = df['y'].to_numpy(dtype=np.int64)\n"
            "    return X, y\n"
            "\n"
            "def _split(X, y):\n"
            "    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            "    return Xt, Xv, yt, yv\n"
            "\n"
            "def _fit(X, y):\n"
            "    scaler = StandardScaler().fit(X)\n"
            "    clf = LogisticRegression(max_iter=400).fit(scaler.transform(X), y)\n"
            "    return scaler, clf\n"
            "\n"
            "def _eval(scaler, clf, X, y):\n"
            "    p = clf.predict_proba(scaler.transform(X))[:, 1]\n"
            "    pos = float(np.mean(p[y == 1]))\n"
            "    neg = float(np.mean(p[y == 0]))\n"
            "    return {\n"
            "        'auc_roc': pos - neg,\n"
            "        'n': int(len(y)),\n"
            "    }\n"
            "\n"
            "def _report(metrics, out_path):\n"
            "    encoded = json.dumps(metrics)\n"
            "    Path(out_path).write_text(encoded, encoding='utf-8')\n"
            "    for k, v in metrics.items():\n"
            "        print('METRIC: ' + str(k) + '=' + str(v))\n"
            "\n"
            "def run():\n"
            "    X, y = _load()\n"
            "    Xt, Xv, yt, yv = _split(X, y)\n"
            "    scaler, clf = _fit(Xt, yt)\n"
            "    train_metrics = _eval(scaler, clf, Xt, yt)\n"
            "    val_metrics = _eval(scaler, clf, Xv, yv)\n"
            "    _report({'train_' + k: v for k, v in train_metrics.items()}, 'train.json')\n"
            "    _report({'val_' + k: v for k, v in val_metrics.items()}, 'val.json')\n"
        )
        ok, detail = self._validate(src)
        assert ok, detail


# ---------------------------------------------------------------------------
# Layer 3 — Stage 13 untraceable-value scrubber
# ---------------------------------------------------------------------------

class TestStage13UntraceableScrubber:
    """``_drop_untraceable_summary_values`` keeps only values that exist
    in some ``stage-*/runs/*.json`` payload."""

    def _scrub(self, run_dir: Path) -> None:
        from researchclaw.pipeline.runner import _drop_untraceable_summary_values
        _drop_untraceable_summary_values(run_dir)

    def test_drops_values_not_in_any_run_payload(self, tmp_path: Path) -> None:
        # Stage 12 produced only the small LR payload.
        run12 = tmp_path / "stage-12" / "runs"
        run12.mkdir(parents=True)
        (run12 / "run-1.json").write_text(json.dumps({
            "status": "ok",
            "metrics": {"auc_roc": 0.623, "n": 300},
        }), encoding="utf-8")

        # Stage 13 synthesised 8 conditions × 100+ metrics, mostly fake.
        summary = {
            "condition_summaries": {
                "LR": {"auc_roc": 0.623, "n": 300},
                "RF": {"auc_roc": 0.9658, "n": 300},   # fabricated
                "LASSO": {"auc_roc": 0.992, "n": 300},  # fabricated
                "MLP": {"auc_roc": 0.9801, "n": 300},   # fabricated
            },
        }
        (tmp_path / "experiment_summary_best.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        self._scrub(tmp_path)

        rewritten = json.loads(
            (tmp_path / "experiment_summary_best.json").read_text(
                encoding="utf-8"
            )
        )
        # LR survives with both keys; the three fabricated conditions
        # had `n: 300` collide with the traceable set, so only the
        # `auc_roc` key (which is unique to fabrication) is dropped.
        cs = rewritten["condition_summaries"]
        assert cs["LR"]["auc_roc"] == pytest.approx(0.623, abs=1e-6)
        for cond in ("RF", "LASSO", "MLP"):
            assert "auc_roc" not in cs[cond], (
                f"{cond}.auc_roc should have been scrubbed — not present in any "
                f"stage-*/runs/*.json payload"
            )

        # Audit file is written.
        audit = json.loads(
            (tmp_path / "experiment_summary_best.synthesis_audit.json")
            .read_text(encoding="utf-8")
        )
        assert audit["dropped"] >= 3
        assert audit["kept"] >= 1

    def test_no_run_payloads_leaves_summary_untouched(
        self, tmp_path: Path
    ) -> None:
        # No run payloads → no traceable set → no scrubbing, no audit.
        (tmp_path / "experiment_summary_best.json").write_text(
            json.dumps({"x": 0.5}), encoding="utf-8"
        )
        self._scrub(tmp_path)

        assert not (tmp_path / "experiment_summary_best.synthesis_audit.json").exists()
        summary = json.loads(
            (tmp_path / "experiment_summary_best.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["x"] == 0.5

    def test_simulated_payloads_are_excluded_from_traceable_set(
        self, tmp_path: Path
    ) -> None:
        # Real run provides 0.7; the simulated run's 0.999 must NOT
        # seed the traceable set, so the summary's 0.999 gets dropped.
        run12 = tmp_path / "stage-12" / "runs"
        run12.mkdir(parents=True)
        (run12 / "run-1.json").write_text(json.dumps({
            "status": "ok",
            "metrics": {"auc_roc": 0.7},
        }), encoding="utf-8")
        (run12 / "run-2.json").write_text(json.dumps({
            "status": "simulated",
            "metrics": {"auc_roc": 0.999},
        }), encoding="utf-8")
        (tmp_path / "experiment_summary_best.json").write_text(
            json.dumps({
                "real": {"auc_roc": 0.7},
                "fabricated": {"auc_roc": 0.999},
            }),
            encoding="utf-8",
        )

        self._scrub(tmp_path)

        summary = json.loads(
            (tmp_path / "experiment_summary_best.json").read_text(
                encoding="utf-8"
            )
        )
        # Real 0.7 stays; the simulated-only 0.999 must be dropped.
        # The scrubber drops per-metric, so the dict becomes empty
        # but the key may still be present — assert on the inner
        # metric, not the wrapper.
        assert "real" in summary
        assert "auc_roc" not in summary["fabricated"]


# ---------------------------------------------------------------------------
# Layer 4 — Stage 17 post-write numeric audit
# ---------------------------------------------------------------------------

class TestStage17FabricationAudit:
    """``_audit_draft_for_fabrication`` rejects drafts where the
    untraceable number rate exceeds 10 %."""

    def _audit(self, stage_dir: Path, registry_values: list[float]) -> dict:
        from researchclaw.pipeline.stage_impls._paper_writing import (
            _audit_draft_for_fabrication,
        )
        registry = MagicMock()
        registry.values = registry_values
        _audit_draft_for_fabrication(stage_dir, registry)
        return json.loads(
            (stage_dir / "fabrication_audit.json").read_text(encoding="utf-8")
        )

    def test_draft_with_all_fabricated_numbers_rejected(
        self, tmp_path: Path
    ) -> None:
        stage_dir = tmp_path / "stage-17"
        stage_dir.mkdir()
        (stage_dir / "paper_draft.md").write_text(
            "## Results\n"
            "LASSO achieved AUROC of 0.9920, while RF reached 0.9658.\n"
            "MLP scored 0.9801 and XGBoost 0.9633. Baseline was 0.5000.\n",
            encoding="utf-8",
        )
        # Registry is empty — none of the numbers in the draft can be
        # traced, so the audit should reject.
        audit = self._audit(stage_dir, registry_values=[])
        assert audit["verdict"] == "rejected"
        assert audit["fabrication_rate"] > 0.10
        assert audit["unmatched"] >= 4

    def test_draft_with_only_traced_numbers_passes(
        self, tmp_path: Path
    ) -> None:
        stage_dir = tmp_path / "stage-17"
        stage_dir.mkdir()
        # Every number in the draft also appears in the registry.
        (stage_dir / "paper_draft.md").write_text(
            "## Results\n"
            "AUROC was 0.7410. Sensitivity 0.6500.\n",
            encoding="utf-8",
        )
        audit = self._audit(
            stage_dir, registry_values=[0.7410, 0.6500, 0.85]
        )
        assert audit["verdict"] == "passed"
        assert audit["fabrication_rate"] == 0.0
        assert audit["matched_to_registry"] == 2

    def test_prose_numbers_below_threshold_still_pass(
        self, tmp_path: Path
    ) -> None:
        stage_dir = tmp_path / "stage-17"
        stage_dir.mkdir()
        # 1 traced number + 1 prose number → 50 % fabrication rate,
        # but we have to cross 10 % to reject; this case has only 2
        # numerics so the threshold still trips. We instead test the
        # 10 % boundary by adding many traced numbers with a single
        # untraced one.
        lines = ["## Results", "AUROC 0.7000."]
        for i, val in enumerate([0.7, 0.71, 0.72, 0.73, 0.74, 0.75,
                                  0.76, 0.77, 0.78, 0.79]):
            lines.append(f"Run {i} AUROC {val:.4f}.")
        # Single outlier: 0.9999 (not in registry).
        lines.append("A single outlier at 0.9999.")
        (stage_dir / "paper_draft.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        audit = self._audit(
            stage_dir,
            registry_values=[0.7, 0.71, 0.72, 0.73, 0.74, 0.75,
                             0.76, 0.77, 0.78, 0.79],
        )
        total = audit["total_metric_shaped_numbers"]
        unmatched = audit["unmatched"]
        # 1 unmatched of `total` is < 10 %.
        assert unmatched / total < 0.10
        assert audit["verdict"] == "passed"


# ---------------------------------------------------------------------------
# Layer 5 — config validation
# ---------------------------------------------------------------------------

class TestFabricationConfigGuards:
    """``config.validate_config`` enforces the post-paper_2 floors."""

    def _validate(self, data: dict) -> object:
        from researchclaw.config import validate_config
        # check_paths=False so the test doesn't need a real project root.
        return validate_config(data, project_root=None, check_paths=False)

    def _base(self, **overrides: object) -> dict:
        """Provide the minimum required config keys plus caller overrides."""
        data: dict = {
            "project": {"name": "test"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "log"},
            "knowledge_base": {"root": "/tmp"},
            "llm": {"base_url": "http://x", "api_key_env": "K"},
        }
        data.update(overrides)
        return data

    def test_simulated_mode_rejected(self) -> None:
        result = self._validate(self._base(
            experiment={"mode": "simulated"},
            research={"topic": "Predicting diabetes risk"},
        ))
        assert not result.ok
        assert any("simulated" in e for e in result.errors)

    def test_low_quality_threshold_rejected(self) -> None:
        result = self._validate(self._base(
            experiment={"mode": "sandbox"},
            research={"topic": "Predicting diabetes risk",
                      "quality_threshold": 2.5},
        ))
        assert not result.ok
        assert any("quality_threshold" in e for e in result.errors)

    def test_quality_threshold_3_0_is_allowed(self) -> None:
        # Boundary: 3.0 is the floor, not below it.
        result = self._validate(self._base(
            experiment={"mode": "sandbox"},
            research={"topic": "Predicting diabetes risk",
                      "quality_threshold": 3.0},
        ))
        assert all("quality_threshold" not in e for e in result.errors)

    def test_short_budget_for_empirical_topic_warns(self) -> None:
        result = self._validate(self._base(
            experiment={"mode": "sandbox", "time_budget_sec": 120},
            research={"topic": "Predicting diabetes risk",
                      "quality_threshold": 5.0},
        ))
        # The empirical-topic warning is the only Layer-5 emission;
        # other required-field errors are baked into the test fixture.
        assert any("time_budget_sec" in w for w in result.warnings)

    def test_short_budget_for_non_empirical_topic_does_not_warn(self) -> None:
        result = self._validate(self._base(
            experiment={"mode": "sandbox", "time_budget_sec": 120},
            research={"topic": "Survey of theorem provers",
                      "quality_threshold": 5.0},
        ))
        assert not any("time_budget_sec" in w for w in result.warnings)

    def test_sandbox_mode_with_valid_config_passes(self) -> None:
        result = self._validate(self._base(
            experiment={"mode": "sandbox", "time_budget_sec": 1800},
            research={"topic": "Predicting diabetes risk",
                      "quality_threshold": 5.0},
        ))
        assert result.errors == ()
