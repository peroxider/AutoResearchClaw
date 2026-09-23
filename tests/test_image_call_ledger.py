"""Round 49: Stage 22 image-model attempts reach run-level cost accounting."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from researchclaw.llm.image_call_ledger import (
    build_image_call_ledger,
    collect_image_call_records,
    image_call_totals,
    record_image_call,
    reset_image_call_ledger,
    validate_image_call_ledger,
    write_image_call_ledger,
)
from researchclaw.pipeline.evidence_store import (
    EvidenceKey,
    EvidenceRecord,
    EvidenceStore,
    content_hash,
    file_hash,
)
from researchclaw.pipeline.final_acceptance import (
    PRESENTATION_DIMENSIONS,
    RESEARCH_DIMENSIONS,
    assess_delivery,
    compilation_inputs,
    inventory,
)
from researchclaw.pipeline.resource_ledger import (
    ResourceLedgerError,
    build_resource_ledger,
    validate_resource_ledger,
)
from tests.test_final_acceptance import write_json
from tests.test_framework_diagram import _PNG_BYTES, _config_with_framework_diagram


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_image_call_ledger()
    yield
    reset_image_call_ledger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(**overrides):
    record = {"provider": "fake", "model": "m1", "endpoint": "https://fake.example/v1",
              "mode": "prompt", "status": "succeeded", "error_type": None,
              "duration_ms": 12, "image_bytes": 7, "usage": None}
    record.update(overrides)
    return record


def _write_delivery(tmp_path: Path) -> Path:
    """Minimal bundle that assess_delivery can fully evaluate."""
    root = tmp_path
    (root / "paper.tex").write_text(
        r"\section{Results} Baseline achieves 81.00. \cite{smith2024}", encoding="utf-8")
    (root / "paper_final.md").write_text(
        "# Results\nBaseline achieves 81.00. [smith2024]", encoding="utf-8")
    (root / "references.bib").write_text(
        "@article{smith2024,title={A paper}}", encoding="utf-8")
    (root / "paper.pdf").write_bytes(b"%PDF-1.4\nfixture")
    (root / "raw.csv").write_text("label,prediction\n1,1\n", encoding="utf-8")
    (root / "source.txt").write_text("Baseline reference evidence.", encoding="utf-8")
    key = EvidenceKey("data", "v1", "test", "Baseline", "config", "42", "accuracy", "mean")
    store = EvidenceStore()
    result_id = store.add(EvidenceRecord(
        key, 81.0, "percent", "success", "commit", "env", "run", "evaluator", True,
        (("raw.csv", file_hash(root / "raw.csv")),)))
    write_json(root, "evidence_store.json", store.to_dict())
    write_json(root, "experiment_protocol.json", {"required_keys": [asdict(key)]})
    papers = {name: file_hash(root / name) for name in ("paper.tex", "paper_final.md")}
    spans = {}
    for name, body in papers.items():
        text = (root / name).read_text(encoding="utf-8")
        start = text.index("81.00")
        spans[name] = {"start": start, "end": start + 5}
    write_json(root, "numeric_claims.json", {
        "evidence_version": store.version, "manuscript_hashes": papers,
        "claims": [{"result_id": result_id, "key": asdict(key), "value": 81.0,
                    "unit": "percent", "decimals": 2, "rendered": "81.00",
                    "spans": spans}]})
    write_json(root, "citation_support.json", {
        "manuscript_hashes": papers,
        "citations": [{"cite_key": "smith2024", "status": "verified",
                       "checker": "fixture-expert", "source": "source.txt",
                       "source_sha256": file_hash(root / "source.txt"),
                       "locator": "p.1", "excerpt": "Baseline reference evidence.",
                       "claim": "Baseline achieves 81.00."}]})
    write_json(root, "quality_report.json", {"score_1_to_10": 8})
    write_json(root, "verification_report.json", {"status": "verified"})
    write_json(root, "compilation.json", {
        "success": True, "inputs": compilation_inputs(root),
        "pdf_sha256": file_hash(root / "paper.pdf")})
    return root


def _bind_review(root: Path) -> None:
    write_json(root, "final_reviews.json", {
        "input_version": content_hash(inventory(root)),
        "dimensions": {d: {"status": "not_applicable" if d == "theory" else "passed",
                           "checker": "fixture-expert", "evidence": "fixture-review-log"}
                       for d in (*RESEARCH_DIMENSIONS, *PRESENTATION_DIMENSIONS)},
    })


class _GoodProvider:
    name = "ledger_good"

    def describe(self):
        return {"provider": self.name, "model": "fake-model-9",
                "endpoint": "https://fake.example/v1/images/generations"}

    def generate(self, prompt, *, aspect_ratio, size):
        return _PNG_BYTES


class _FailingProvider:
    name = "ledger_bad"

    def generate(self, prompt, *, aspect_ratio, size):
        raise RuntimeError("nope")


def _generate_diagram_into(root: Path, providers) -> None:
    """Run the real Stage 22 orchestrator with fake providers."""
    from researchclaw.agents.figure_agent import framework_diagram as fd

    cfg = _config_with_framework_diagram(provider="auto", render_mode="direct")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            fd, "_generate_framework_diagram_prompt",
            lambda paper_text, config, llm: (
                "# Framework Diagram Prompt\n"
                "## Image Generation Prompt\n"
                "X\n"
            ),
            raising=False,
        )
        monkeypatch.setattr(fd, "build_framework_diagram_providers",
                            lambda **kw: providers)
        artifacts, png_path = fd.generate_framework_diagram_artifacts(
            paper_text="# Test\n", config=cfg, output_dir=root / "charts", llm=None)
    finally:
        monkeypatch.undo()
    assert png_path is not None and png_path.is_file()


# ---------------------------------------------------------------------------
# Ledger unit behavior
# ---------------------------------------------------------------------------


def test_ledger_roundtrip_totals_and_version() -> None:
    records = [_record(), _record(status="failed", error_type="RuntimeError",
                                  image_bytes=None, model=None, endpoint=None)]
    ledger = build_image_call_ledger(records)
    validate_image_call_ledger(ledger)
    assert ledger["totals"] == {"calls": 2, "failures": 1, "total_image_bytes": 7}
    assert collect_image_call_records() == []


def test_ledger_rejects_tampering() -> None:
    ledger = build_image_call_ledger([_record()])
    validate_image_call_ledger(ledger)
    for mutation in (
        {"totals": {"calls": 1, "failures": 0, "total_image_bytes": 99}},
        {"calls": []},
        {"schema_version": 2},
    ):
        broken = dict(ledger)
        broken.update(mutation)
        with pytest.raises(ValueError):
            validate_image_call_ledger(broken)


@pytest.mark.parametrize("mutation", [
    {"provider": ""}, {"provider": 7}, {"mode": "generate"},
    {"status": "raised"}, {"status": "failed", "error_type": None},
    {"status": "succeeded", "error_type": "RuntimeError"},
    {"duration_ms": True}, {"duration_ms": -1}, {"duration_ms": None},
    {"image_bytes": True}, {"image_bytes": -3},
    {"usage": ["tokens"]}, {"model": ""}, {"endpoint": 3},
])
def test_ledger_rejects_malformed_records(mutation) -> None:
    with pytest.raises(ValueError):
        build_image_call_ledger([_record(**mutation)])


def test_totals_follow_the_token_total_precedent() -> None:
    # Mixed null/present sizes sum the reported ones; all-null stays null.
    assert image_call_totals([_record(image_bytes=None),
                              _record(status="failed", error_type="RuntimeError",
                                      image_bytes=None)]) == {
        "calls": 2, "failures": 1, "total_image_bytes": None}
    assert image_call_totals([_record(image_bytes=None), _record()]) == {
        "calls": 2, "failures": 0, "total_image_bytes": 7}


def test_write_image_call_ledger_writes_and_clears(tmp_path: Path) -> None:
    assert write_image_call_ledger(tmp_path) is None
    assert not (tmp_path / "image_call_ledger.json").exists()
    record_image_call(_record())
    ledger = write_image_call_ledger(tmp_path)
    assert ledger is not None
    frozen = json.loads((tmp_path / "image_call_ledger.json").read_text("utf-8"))
    assert frozen == ledger
    assert collect_image_call_records() == []
    validate_image_call_ledger(frozen)


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


def test_orchestrator_attempts_reach_the_run_level_ledger(tmp_path: Path) -> None:
    _generate_diagram_into(tmp_path, [_FailingProvider(), _GoodProvider()])
    records = collect_image_call_records()
    assert [record["status"] for record in records] == ["failed", "succeeded"]
    failed, good = records
    assert failed["provider"] == "ledger_bad" and failed["error_type"] == "RuntimeError"
    assert failed["image_bytes"] is None and failed["model"] is None
    assert failed["mode"] == "prompt" and type(failed["duration_ms"]) is int
    assert failed["duration_ms"] >= 0
    assert good["provider"] == "ledger_good"
    assert good["model"] == "fake-model-9"
    assert good["endpoint"] == "https://fake.example/v1/images/generations"
    assert good["image_bytes"] == len(_PNG_BYTES)
    assert good["usage"] is None
    validate_image_call_ledger(build_image_call_ledger(records))
    manifest = json.loads(
        (tmp_path / "charts" / "framework_diagram_generation.json").read_text("utf-8"))
    assert manifest["generation_attempts"] == [
        {"provider": "ledger_bad", "mode": "prompt", "status": "failed",
         "error_type": "RuntimeError"},
        {"provider": "ledger_good", "mode": "prompt", "status": "succeeded"},
    ]


def test_provider_usage_block_flows_into_the_ledger(tmp_path: Path) -> None:
    class _UsageProvider(_GoodProvider):
        def generate(self, prompt, *, aspect_ratio, size):
            self.last_usage = {"input_tokens": 10, "output_tokens": 4}
            return _PNG_BYTES

    _generate_diagram_into(tmp_path, [_UsageProvider()])
    (record,) = collect_image_call_records()
    assert record["usage"] == {"input_tokens": 10, "output_tokens": 4}


def test_failed_attempt_does_not_inherit_the_previous_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    # A failed attempt must not carry the usage block the endpoint reported
    # on an earlier call: usage is per-call, attributed at record time.
    import time as _time
    from urllib.error import URLError

    from researchclaw.agents.figure_agent import framework_diagram as fd

    provider = fd.OpenAICompatibleProvider(base_url="https://fake.example/v1",
                                           api_key="fixture", model="fake-model")
    payloads = [json.dumps({"usage": {"total_tokens": 7}, "data": []}).encode("utf-8")]

    class _Response:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args) -> bool:
            return False

    def urlopen(request, timeout=None):
        if payloads:
            return _Response(payloads.pop(0))
        raise URLError("transport down")

    monkeypatch.setattr(fd.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError):
        provider.generate("p", aspect_ratio="1:1", size="1024x1024")
    assert provider.last_usage == {"total_tokens": 7}
    with pytest.raises(URLError):
        provider.generate("p", aspect_ratio="1:1", size="1024x1024")
    fd._record_image_attempt(provider, "prompt", _time.monotonic(), None,
                             URLError("transport down"))
    assert collect_image_call_records()[-1]["usage"] is None


def test_hybrid_reference_retry_produces_two_ledger_records(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent import framework_diagram as fd

    class _ReferenceRejectingProvider:
        name = "grsai_gpt_images"

        def generate_with_reference(self, *args, **kwargs):
            raise RuntimeError("reference rejected")

        def generate(self, prompt, *, aspect_ratio, size):
            # The compose step PIL-loads the candidate, so return the real
            # rendered semantic layer like the framework-diagram suite does.
            return self.valid_png

    provider = _ReferenceRejectingProvider()
    cfg = _config_with_framework_diagram(provider="grsai_gpt_images",
                                         render_mode="hybrid")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(fd, "build_framework_diagram_providers",
                            lambda **kw: [provider])
        original = fd._render_semantic_skeleton

        def capture_renderer(*args, **kwargs):
            result = original(*args, **kwargs)
            provider.valid_png = result[2].read_bytes()
            return result

        monkeypatch.setattr(fd, "_render_semantic_skeleton", capture_renderer)
        artifacts, png_path = fd.generate_framework_diagram_artifacts(
            paper_text="# Test\nInput data model output", config=cfg,
            output_dir=tmp_path, llm=None)
    finally:
        monkeypatch.undo()
    assert png_path is not None
    records = collect_image_call_records()
    assert [(record["mode"], record["status"]) for record in records] == [
        ("reference", "failed"), ("text_free", "succeeded")]
    assert records[0]["error_type"] == "RuntimeError"
    manifest = json.loads((tmp_path / "framework_diagram_generation.json")
                          .read_text("utf-8"))
    assert manifest["generation_attempts"] == [
        {"provider": "grsai_gpt_images", "mode": "reference", "status": "failed",
         "error_type": "RuntimeError"},
        {"provider": "grsai_gpt_images", "mode": "text_free", "status": "succeeded"},
    ]


def test_hybrid_total_provider_failure_keeps_the_provider_mode(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent import framework_diagram as fd

    class _DoomedHybridProvider:
        name = "grsai_gpt_images"

        def generate_with_reference(self, *args, **kwargs):
            raise RuntimeError("reference rejected")

        def generate(self, prompt, *, aspect_ratio, size):
            raise RuntimeError("gateway down")

    cfg = _config_with_framework_diagram(render_mode="hybrid")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(fd, "build_framework_diagram_providers",
                            lambda **kw: [_DoomedHybridProvider()])
        fd.generate_framework_diagram_artifacts(
            paper_text="# Test\nInput data model output", config=cfg,
            output_dir=tmp_path, llm=None)
    finally:
        monkeypatch.undo()
    records = collect_image_call_records()
    assert [(record["mode"], record["status"]) for record in records] == [
        ("reference", "failed"), ("provider", "failed")]
    assert all(record["image_bytes"] is None and record["model"] is None
               for record in records)
    ledger = write_image_call_ledger(tmp_path)
    assert ledger is not None
    validate_image_call_ledger(ledger)
    assert ledger["totals"] == {"calls": 2, "failures": 2,
                                "total_image_bytes": None}


def test_falsy_provider_return_records_no_attempt(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent import framework_diagram as fd

    class _EmptyProvider:
        name = "empty"

        def generate(self, prompt, *, aspect_ratio, size):
            return b""

    cfg = _config_with_framework_diagram(provider="auto", render_mode="direct")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            fd, "_generate_framework_diagram_prompt",
            lambda paper_text, config, llm: (
                "# Framework Diagram Prompt\n## Image Generation Prompt\nX\n"),
            raising=False,
        )
        monkeypatch.setattr(fd, "build_framework_diagram_providers",
                            lambda **kw: [_EmptyProvider()])
        fd.generate_framework_diagram_artifacts(
            paper_text="# Test\n", config=cfg, output_dir=tmp_path, llm=None)
    finally:
        monkeypatch.undo()
    assert collect_image_call_records() == []
    assert write_image_call_ledger(tmp_path) is None


# ---------------------------------------------------------------------------
# Resource-ledger aggregation
# ---------------------------------------------------------------------------


def test_resource_ledger_aggregates_image_calls(tmp_path: Path) -> None:
    ledger = build_image_call_ledger([
        _record(), _record(status="failed", error_type="RuntimeError",
                           image_bytes=None, model=None, endpoint=None)])
    (tmp_path / "image_call_ledger.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8")
    built = build_resource_ledger(tmp_path)
    (row,) = [row for row in built["rows"]
              if row["source"] == "image_call_ledger.json"]
    assert row["kind"] == "image_generation"
    assert row["calls"] == 2 and row["failures"] == 1
    assert row["image_bytes"] == 7 and row["version"] == ledger["version"]
    assert built["errors"] == []
    validate_resource_ledger(tmp_path, built)


def test_resource_ledger_fails_closed_on_tampered_image_ledger(tmp_path: Path) -> None:
    ledger = build_image_call_ledger([_record()])
    ledger["totals"]["calls"] = 99
    (tmp_path / "image_call_ledger.json").write_text(
        json.dumps(ledger), encoding="utf-8")
    built = build_resource_ledger(tmp_path)
    # The version hash covers totals, so the frozen aggregate is rejected as
    # changed — either way the source fails closed.
    assert any(error["source"] == "image_call_ledger.json"
               for error in built["errors"])
    from researchclaw.pipeline.resource_ledger import ledger_issues
    assert "unreadable_resource_source:image_call_ledger.json" in ledger_issues(built)


def test_resource_ledger_rejects_forged_aggregate(tmp_path: Path) -> None:
    ledger = build_image_call_ledger([_record()])
    (tmp_path / "image_call_ledger.json").write_text(
        json.dumps(ledger), encoding="utf-8")
    built = build_resource_ledger(tmp_path)
    built["rows"] = []
    with pytest.raises(ResourceLedgerError):
        validate_resource_ledger(tmp_path, built)


# ---------------------------------------------------------------------------
# Acceptance cross-check: manifest attempts must be ledger-covered
# ---------------------------------------------------------------------------


def _issue_reasons(report: dict) -> list[str]:
    return [issue["reason"] for issue in report["issues"]]


def test_acceptance_requires_ledger_for_provider_attempts(tmp_path: Path) -> None:
    root = _write_delivery(tmp_path)
    _generate_diagram_into(root, [_GoodProvider()])
    _bind_review(root)
    report = assess_delivery(root, target_status="research_complete")
    assert "image_model_calls_missing_run_ledger" in _issue_reasons(report)
    assert report["dimensions"]["resources"] == "failed"


def test_acceptance_accepts_a_contained_ledger(tmp_path: Path) -> None:
    root = _write_delivery(tmp_path)
    _generate_diagram_into(root, [_GoodProvider()])
    assert write_image_call_ledger(root) is not None
    _bind_review(root)
    report = assess_delivery(root, target_status="research_complete")
    reasons = _issue_reasons(report)
    assert "image_model_calls_missing_run_ledger" not in reasons
    assert "image_model_calls_not_fully_ledgered" not in reasons
    assert "invalid_image_call_ledger" not in reasons


def test_acceptance_rejects_an_incomplete_ledger(tmp_path: Path) -> None:
    root = _write_delivery(tmp_path)
    _generate_diagram_into(root, [_FailingProvider(), _GoodProvider()])
    records = collect_image_call_records()
    stripped = build_image_call_ledger(records[1:])
    (root / "image_call_ledger.json").write_text(
        json.dumps(stripped, indent=2), encoding="utf-8")
    _bind_review(root)
    report = assess_delivery(root, target_status="research_complete")
    assert "image_model_calls_not_fully_ledgered" in _issue_reasons(report)
    assert report["dimensions"]["resources"] == "failed"


def test_acceptance_rejects_an_invalid_image_ledger(tmp_path: Path) -> None:
    root = _write_delivery(tmp_path)
    _generate_diagram_into(root, [_GoodProvider()])
    (root / "image_call_ledger.json").write_text("{not json", encoding="utf-8")
    _bind_review(root)
    report = assess_delivery(root, target_status="research_complete")
    assert "invalid_image_call_ledger" in _issue_reasons(report)


def test_acceptance_ignores_matplotlib_only_manifests(tmp_path: Path) -> None:
    root = _write_delivery(tmp_path)
    charts = root / "charts"
    charts.mkdir()
    write_json(charts, "framework_diagram_generation.json", {
        "schema_version": 2, "artifact": "framework_diagram.png",
        "render_mode": "direct", "provider": "matplotlib",
        "generation_attempts": [], "model": None, "size": "1536x1024",
        "aspect_ratio": "3:2", "prompt_sha256": "0" * 64,
        "prompt_artifact": "framework_diagram_image_prompt.txt",
        "original_output": None, "original_output_sha256": None,
        "image_sha256": "0" * 64})
    _bind_review(root)
    report = assess_delivery(root, target_status="research_complete")
    reasons = _issue_reasons(report)
    assert not any(reason.startswith("image_model_calls_") or
                   reason == "invalid_image_call_ledger" for reason in reasons)
