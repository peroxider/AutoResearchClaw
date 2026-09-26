from __future__ import annotations

import copy
import shutil

import pytest

from researchclaw.experiment.protocol_runner import run_matrix
from researchclaw.experiment.reproduction_comparison import (
    ReproductionComparisonError, compare_reproduction_bundles, compare_reproduction_values,
    verify_reproduction_report,
)
from researchclaw.pipeline.evidence_store import content_hash
from tests.test_experiment_protocol import spec
from tests.test_protocol_runner import SCRIPT, setup
from tests.test_research_inputs import inputs


def _independent_inputs(inputs, destination):
    source = destination / "source"
    shutil.copytree(inputs[0].parent, source)
    return (source / "brief.json", destination / "run",
            copy.deepcopy(inputs[2]), copy.deepcopy(inputs[3]))


def _completed_pair(inputs, spec, tmp_path, right_script=SCRIPT, left_script=SCRIPT):
    left_root, left_project, _, left_cfg = setup(
        inputs, copy.deepcopy(spec), left_script)
    assert run_matrix(left_root, left_project, left_cfg.experiment)["status"] == "complete"
    right_inputs = _independent_inputs(inputs, tmp_path / "second-site")
    right_root, right_project, _, right_cfg = setup(
        right_inputs, copy.deepcopy(spec), right_script)
    assert run_matrix(right_root, right_project, right_cfg.experiment)["status"] == "complete"
    return left_root, right_root


def test_two_independent_complete_bundles_match_exact_metrics(inputs, spec, tmp_path):
    left, right = _completed_pair(inputs, spec, tmp_path)
    output = tmp_path / "reproduction.json"
    report = compare_reproduction_bundles(
        left, right, left_site="site-a", right_site="site-b", output=output)
    assert report["status"] == "matched", {
        key: report[key] for key in ("identities_match", "execution_evidence_distinct",
                                    "missing_left", "missing_right", "mismatches",
                                    "record_count", "sites")}
    assert report["record_count"] == 8
    assert report["identities_match"] is True
    assert report["execution_evidence_distinct"] is True
    assert report["environment_relation"] == "same"
    assert all(pair["value_match"] for pair in report["pairs"])
    assert all(pair["allowed_difference"] == 0 for pair in report["pairs"])
    assert verify_reproduction_report(report, left, right) == report
    assert output.is_file()


def test_a_copied_bundle_is_not_independent_execution_evidence(inputs, spec, tmp_path):
    root, project, _, cfg = setup(inputs, spec)
    assert run_matrix(root, project, cfg.experiment)["status"] == "complete"
    copied = tmp_path / "copied-run"
    shutil.copytree(root, copied)
    report = compare_reproduction_bundles(
        root, copied, left_site="original", right_site="copy")
    assert report["status"] == "mismatched"
    assert report["execution_evidence_distinct"] is False
    assert all(pair["value_match"] for pair in report["pairs"])


def test_changed_source_identity_cannot_be_reported_as_reproduction(inputs, spec, tmp_path):
    changed = SCRIPT + "\n# independently changed source\n"
    left, right = _completed_pair(inputs, spec, tmp_path, changed)
    report = compare_reproduction_bundles(
        left, right, left_site="site-a", right_site="site-b")
    assert report["status"] == "mismatched"
    assert report["identities_match"] is False
    assert report["protocol_version"]


def test_self_consistent_report_tamper_is_recomputed_from_bundles(inputs, spec, tmp_path):
    left, right = _completed_pair(inputs, spec, tmp_path)
    report = compare_reproduction_bundles(
        left, right, left_site="site-a", right_site="site-b")
    report["status"] = "mismatched"
    report["version"] = content_hash({key: value for key, value in report.items()
                                      if key != "version"})
    with pytest.raises(ReproductionComparisonError, match="differs"):
        verify_reproduction_report(report, left, right)


def test_predeclared_tolerance_accepts_bounded_cross_run_difference(inputs, spec, tmp_path):
    spec["reproduction"] = {"metrics": {"accuracy": {
        "absolute_tolerance": 0.5,
        "relative_tolerance": 0,
        "rationale": "Fixture allowance declared before either matrix executes.",
    }}}
    site_script = SCRIPT.replace(
        'writer.writerows([row[data["id_column"]], data["label_encoding"][label]] for row in test)',
        'writer.writerows([row[data["id_column"]], '
        '(int(row["row"][1:]) % 2 if "second-site" in str(Path.cwd()) '
        'else data["label_encoding"][label])] for row in test)')
    left, right = _completed_pair(
        inputs, spec, tmp_path, right_script=site_script, left_script=site_script)
    report = compare_reproduction_bundles(
        left, right, left_site="site-a", right_site="site-b")
    assert report["status"] == "matched"
    assert report["reproduction_policy"]["mode"] == "tolerance"
    assert any(pair["absolute_difference"] == 0.5 for pair in report["pairs"])
    assert all(pair["value_match"] for pair in report["pairs"])


def test_value_comparison_uses_symmetric_predeclared_rule():
    exact = {"mode": "exact", "metrics": {}}
    tolerant = {"mode": "tolerance", "metrics": {"mse": {
        "absolute_tolerance": 0.1, "relative_tolerance": 0.1, "rationale": "fixture"}}}
    exact_match, exact_difference, exact_allowed = compare_reproduction_values(
        exact, "mse", 1.0, 1.01)
    assert exact_match is False and exact_difference == pytest.approx(0.01)
    assert exact_allowed == 0
    matched, difference, allowed = compare_reproduction_values(tolerant, "mse", 1.0, 1.21)
    assert matched is True and difference == pytest.approx(0.21)
    assert allowed == pytest.approx(0.221)
    assert compare_reproduction_values(
        exact, "mse", -1.7e308, 1.7e308) == (False, None, 0.0)


@pytest.mark.parametrize(("left", "right"), [
    ("same", "same"),
    ("../escape", "safe"),
    ("", "safe"),
    (None, "safe"),
])
def test_site_labels_are_bounded_self_declared_identifiers(tmp_path, left, right):
    with pytest.raises(ReproductionComparisonError, match="Site labels"):
        compare_reproduction_bundles(
            tmp_path / "left", tmp_path / "right", left_site=left, right_site=right)
