from __future__ import annotations

import json
import zipfile

import pytest

from researchclaw.pipeline.benchmark_distribution import (
    BenchmarkDistributionError, build_runner_bundle, extract_runner_bundle,
    verify_runner_bundle,
)
from researchclaw.pipeline.benchmark_runner import run_benchmark_suite
from researchclaw.pipeline.benchmark_suite import verify_public_benchmark_suite
from tests.test_benchmark_runner import sealed_runner_fixture


def test_runner_bundle_contains_only_explicit_public_and_sealed_inputs(
        tmp_path, monkeypatch):
    (public, plan_path, gold_path, suite, curator_key, runner_key,
     assessor_key, evaluator_key) = sealed_runner_fixture(tmp_path, monkeypatch)
    (public / "unrelated-secret.pem").write_text("secret", encoding="utf-8")
    bundle = tmp_path / "runner.zip"
    manifest = build_runner_bundle(
        suite_report=suite, public_root=public, plan_path=plan_path,
        output_path=bundle)
    assert verify_runner_bundle(bundle) == manifest
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        text = "\n".join(
            archive.read(name).decode("utf-8", errors="ignore") for name in names)
    assert "unrelated-secret.pem" not in names
    assert "Private gold rubric" not in text
    assert all(path.name not in names for path in (
        gold_path, curator_key, runner_key, assessor_key, evaluator_key))
    destination = tmp_path / "materialized"
    assert extract_runner_bundle(bundle, destination) == manifest
    extracted_suite = json.loads(
        (destination / "_arc" / "benchmark_suite.json").read_text(encoding="utf-8"))
    assert verify_public_benchmark_suite(
        extracted_suite, destination,
        destination / plan_path.relative_to(public)) == extracted_suite
    gold_path.unlink()
    results = tmp_path / "bundle-results"
    result_manifest = run_benchmark_suite(
        suite_report=extracted_suite, public_root=destination,
        plan_path=destination / plan_path.relative_to(public),
        results_root=results, signing_key_path=runner_key)
    assert len(result_manifest["cases"]) == 16


def test_runner_bundle_rejects_tamper_extra_files_and_unsafe_extraction(
        tmp_path, monkeypatch):
    public, plan_path, _, suite, *_ = sealed_runner_fixture(tmp_path, monkeypatch)
    bundle = tmp_path / "runner.zip"
    build_runner_bundle(
        suite_report=suite, public_root=public, plan_path=plan_path,
        output_path=bundle)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(BenchmarkDistributionError, match="not empty"):
        extract_runner_bundle(bundle, nonempty)
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("undeclared.txt", "extra")
    with pytest.raises(BenchmarkDistributionError, match="undeclared"):
        verify_runner_bundle(bundle)
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.txt", "escape")
    with pytest.raises(BenchmarkDistributionError, match="unsafe"):
        verify_runner_bundle(unsafe)


def test_plaintext_or_unsigned_suite_cannot_be_distributed(tmp_path, monkeypatch):
    from tests.test_benchmark_runner import runner_fixture

    public, plan_path, _, suite = runner_fixture(tmp_path, monkeypatch)
    with pytest.raises(BenchmarkDistributionError, match="requires"):
        build_runner_bundle(
            suite_report=suite, public_root=public, plan_path=plan_path,
            output_path=tmp_path / "runner.zip")
