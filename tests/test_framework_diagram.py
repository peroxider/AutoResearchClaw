"""Tests for the Stage 22 framework-diagram generator.

Covers:
  - OpenAI-compatible provider — POST payload, b64_json + url paths,
    Authorization header, non-image rejection.
  - Gemini provider — wraps NanoBananaAgent, propagates bytes, raises on
    failure.
  - Prompt-text extractor — strips the markdown scaffold.
  - Matplotlib fallback — always produces a real PNG.
  - Orchestrator helper — provider failure → matplotlib fallback; first
    successful provider is used; legacy YAML still loads.
  - Paper placeholder path is unchanged (Stage 17 / Stage 22 contract).
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

from researchclaw.config import RCConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Smallest valid PNG (1x1 transparent)
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xfa\xcf\xc0\x00\x00\x00\x03\x00\x01"
    b"\x86\x1b\x8c\x9c"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _reviewable_png() -> bytes:
    """A provider-shaped PNG that passes the raster publication review.

    ``_PNG_BYTES`` above is signature-valid but not PIL-decodable, so the
    Stage-22 gate rejects it and the orchestrator falls back to matplotlib.
    Orchestrator tests that need the provider's bytes to actually be embedded
    use this 1024x640 diagram-like raster instead.
    """
    import io

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1024, 640), "white")
    draw = ImageDraw.Draw(image)
    for index in range(80):
        shade = index * 3
        draw.rectangle(
            [index * 12 % 984, index * 7 % 600,
             index * 12 % 984 + 30, index * 7 % 600 + 20],
            fill=(shade % 256, (shade * 7) % 256, (shade * 13) % 256))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# Frozen once so byte-equality and sha256 assertions are stable across calls.
_GOOD_PNG_BYTES = _reviewable_png()


def _undersized_reviewable_png() -> bytes:
    """The reviewable diagram shrunk below the resolution floor.

    Content survives the shrink, so ``too_small`` is the only rejection
    reason — orchestrator fall-through tests reject on size, not content.
    """
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(_GOOD_PNG_BYTES)).resize((200, 100))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _minimal_yaml() -> dict[str, object]:
    """Bare-minimum YAML that constructs RCConfig — same shape as paper_2."""
    return yaml.safe_load("""
project:
  name: "test_paper"
  mode: "full-auto"
  profile: "_generic"
research:
  topic: "Test topic"
  domains: ["ml"]
  daily_paper_count: 1
  quality_threshold: 4.0
runtime:
  timezone: "UTC"
notifications:
  channel: "console"
knowledge_base:
  backend: "markdown"
  root: "docs/kb"
llm:
  provider: "k"
  base_url: "https://x/y"
  api_key_env: "X"
  primary_model: "m"
security: {}
export: {}
memory:
  enabled: false
prompts:
  custom_file: ""
  extra_prompts: {}
""")


def _config_with_framework_diagram(**fd_overrides: object) -> RCConfig:
    cfg = RCConfig.from_dict(_minimal_yaml())
    fd = cfg.experiment.framework_diagram
    # Build a new FrameworkDiagramConfig with overrides applied
    from dataclasses import replace
    new_fd = replace(fd, **fd_overrides)
    new_exp = replace(cfg.experiment, framework_diagram=new_fd)
    return replace(cfg, experiment=new_exp)


# ---------------------------------------------------------------------------
# Provider: OpenAI-compatible
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_openai_provider_post_payload_and_b64_decode() -> None:
    """OpenAI provider builds the right POST and decodes b64_json."""
    from researchclaw.agents.figure_agent.framework_diagram import (
        OpenAICompatibleProvider,
    )

    body = json.dumps({"data": [{"b64_json": _b64(_PNG_BYTES)}]}).encode("utf-8")
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(body)

    with mock.patch(
        "researchclaw.agents.figure_agent.framework_diagram.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="dall-e-3",
        )
        result = provider.generate(
            "architecture diagram",
            aspect_ratio="16:9",
            size="1792x1024",
        )

    assert result == _PNG_BYTES
    assert captured["url"] == "https://api.example.com/v1/images/generations"
    assert captured["method"] == "POST"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    payload = captured["body"]
    assert payload["model"] == "dall-e-3"
    assert payload["prompt"] == "architecture diagram"
    assert payload["size"] == "1792x1024"
    assert payload["response_format"] == "b64_json"
    assert payload["n"] == 1


def test_openai_provider_url_response_path() -> None:
    """When b64_json is absent, the provider follows the URL."""
    from researchclaw.agents.figure_agent.framework_diagram import (
        OpenAICompatibleProvider,
    )

    calls: list[tuple[str, str]] = []

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        url = req.full_url
        calls.append((req.get_method(), url))
        if url.endswith("/images/generations"):
            body = json.dumps(
                {"data": [{"url": "https://cdn.example/img.png"}]}
            ).encode("utf-8")
        else:
            body = _PNG_BYTES
        return _FakeResponse(body)

    with mock.patch(
        "researchclaw.agents.figure_agent.framework_diagram.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            api_key="",
            model="dall-e-3",
        )
        result = provider.generate("anything", aspect_ratio="16:9", size="1024x1024")

    assert result == _PNG_BYTES
    methods = [m for m, _ in calls]
    assert methods == ["POST", "GET"]
    assert calls[1][1] == "https://cdn.example/img.png"


def test_openai_provider_no_auth_when_empty_key() -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        OpenAICompatibleProvider,
    )

    body = json.dumps({"data": [{"b64_json": _b64(_PNG_BYTES)}]}).encode("utf-8")
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["headers"] = dict(req.headers)
        return _FakeResponse(body)

    with mock.patch(
        "researchclaw.agents.figure_agent.framework_diagram.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            api_key="",
            model="dall-e-3",
        ).generate("x", aspect_ratio="16:9", size="1024x1024")

    assert "Authorization" not in captured["headers"]


def test_openai_provider_rejects_non_image_bytes() -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        OpenAICompatibleProvider,
    )

    body = json.dumps(
        {"data": [{"b64_json": _b64(b'{"error": "rate limited"}')}]}
    ).encode("utf-8")

    with mock.patch(
        "researchclaw.agents.figure_agent.framework_diagram.urllib.request.urlopen",
        return_value=_FakeResponse(body),
    ):
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            api_key="k",
            model="dall-e-3",
        )
        with pytest.raises(RuntimeError, match="non-image bytes"):
            provider.generate("x", aspect_ratio="16:9", size="1024x1024")


# ---------------------------------------------------------------------------
# Provider: GRSAI GPT-Images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("api_style", "endpoint", "result_key"),
    [
        ("openai", "/v1/images/generations", "data"),
        ("unified", "/v1/api/generate", "results"),
    ],
)
def test_grsai_provider_contract(
    api_style: str, endpoint: str, result_key: str
) -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        GrsaiGPTImagesProvider,
    )

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        if req.full_url == "https://cdn.example/framework.png":
            return _FakeResponse(_PNG_BYTES)
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(json.dumps({
            result_key: [{"url": "https://cdn.example/framework.png"}]
        }).encode("utf-8"))

    with mock.patch(
        "researchclaw.agents.figure_agent.framework_diagram.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        result = GrsaiGPTImagesProvider(
            base_url="https://grsaiapi.com/v1",
            api_key="secret",
            model="gpt-image-2",
            api_style=api_style,
        ).generate("three module pipeline", aspect_ratio="16:9", size="1024x1024")

    assert result == _PNG_BYTES
    assert str(captured["url"]).endswith(endpoint)
    assert captured["headers"]["Authorization"] == "Bearer secret"
    payload = captured["body"]
    assert payload["model"] == "gpt-image-2"
    assert "publication-quality medical-informatics" in payload["prompt"]
    if api_style == "openai":
        assert payload["image"] == []
        assert payload["response_format"] == "url"
    else:
        assert payload["images"] == []
        assert payload["replyType"] == "json"


def test_grsai_reference_generation_sends_locked_skeleton() -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        GrsaiGPTImagesProvider,
    )

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        if req.full_url == "https://cdn.example/hybrid.png":
            return _FakeResponse(_PNG_BYTES)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(json.dumps({
            "data": [{"url": "https://cdn.example/hybrid.png"}]
        }).encode("utf-8"))

    with mock.patch(
        "researchclaw.agents.figure_agent.framework_diagram.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        result = GrsaiGPTImagesProvider(
            base_url="https://grsaiapi.com",
            api_key="secret",
            quality="high",
        ).generate_with_reference(
            "preserve all semantics",
            _PNG_BYTES,
            aspect_ratio="16:9",
            size="1536x1024",
        )

    assert result == _PNG_BYTES
    payload = captured["body"]
    assert payload["quality"] == "high"
    assert len(payload["image"]) == 1
    assert payload["image"][0].startswith("data:image/png;base64,")
    assert "deliberately formal" in payload["prompt"]


def test_grsai_standard_model_omits_auto_quality() -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        GrsaiGPTImagesProvider,
    )

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(json.dumps({
            "data": [{"b64_json": _b64(_PNG_BYTES)}]
        }).encode("utf-8"))

    with mock.patch(
        "researchclaw.agents.figure_agent.framework_diagram.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        GrsaiGPTImagesProvider(
            base_url="https://grsaiapi.com",
            api_key="secret",
            model="gpt-image-2",
            quality="auto",
        ).generate("formal diagram", aspect_ratio="16:9", size="1024x1024")
    assert "quality" not in captured["body"]


# ---------------------------------------------------------------------------
# Provider: Gemini (NanoBananaAgent wrapper)
# ---------------------------------------------------------------------------


class _StubNanoBananaAgent:
    """Minimal stub for NanoBananaAgent.execute() — controlled by tests."""

    def __init__(self, *, success: bool, png_bytes: bytes = _PNG_BYTES) -> None:
        self._success = success
        self._png_bytes = png_bytes
        self.output_dir = Path("/tmp/_unused_framework_diagram")

    class _Result:
        def __init__(self, data: dict) -> None:
            self.data = data

    def execute(self, context: dict) -> "_StubNanoBananaAgent._Result":
        out_dir = Path(context["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / "framework_diagram_test.png"
        if self._success:
            png_path.write_bytes(self._png_bytes)
            return self._Result({
                "generated": [{
                    "success": True,
                    "output_path": str(png_path),
                    "backend": "nano_banana",
                }],
                "count": 1,
            })
        return self._Result({
            "generated": [{
                "success": False,
                "error": "no Gemini API key",
                "backend": "nano_banana",
            }],
            "count": 0,
        })


def test_gemini_provider_with_stubbed_agent(tmp_path: Path) -> None:
    """GeminiProvider delegates to NanoBananaAgent and returns bytes."""
    from researchclaw.agents.figure_agent.framework_diagram import GeminiProvider

    agent = _StubNanoBananaAgent(success=True)
    provider = GeminiProvider(
        gemini_api_key="gk",
        model="gemini-2.5-flash-image",
        output_dir=tmp_path,
        agent=agent,
    )
    result = provider.generate(
        "architecture diagram", aspect_ratio="16:9", size="1024x1024"
    )
    assert result == _PNG_BYTES


def test_gemini_provider_failure_propagates(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent.framework_diagram import GeminiProvider

    agent = _StubNanoBananaAgent(success=False)
    provider = GeminiProvider(
        gemini_api_key="gk",
        model="gemini-2.5-flash-image",
        output_dir=tmp_path,
        agent=agent,
    )
    with pytest.raises(RuntimeError, match="Gemini generation failed"):
        provider.generate(
            "x", aspect_ratio="16:9", size="1024x1024"
        )


# ---------------------------------------------------------------------------
# Prompt-text extractor
# ---------------------------------------------------------------------------


def test_extract_framework_prompt_text_strips_markdown_scaffold() -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        extract_framework_prompt_text,
    )

    md = (
        "# Framework Diagram Prompt\n"
        "\n"
        "**Paper**: foo\n"
        "\n"
        "## Image Generation Prompt\n"
        "\n"
        "Create a clean academic framework diagram for paper foo.\n"
        "\n"
        "## Usage Instructions\n"
        "\n"
        "1. Copy the prompt above into DALL-E.\n"
    )
    body = extract_framework_prompt_text(md)
    assert "Create a clean academic framework diagram" in body
    assert "Usage Instructions" not in body
    assert "Copy the prompt above" not in body


def test_extract_framework_prompt_text_handles_missing_heading() -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        extract_framework_prompt_text,
    )
    assert extract_framework_prompt_text("just text") == "just text"
    assert extract_framework_prompt_text("") == ""


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def test_provider_construction_explicit_openai_only(tmp_path: Path) -> None:
    """provider='openai_compatible' returns only the OpenAI provider."""
    from researchclaw.agents.figure_agent.framework_diagram import (
        build_framework_diagram_providers,
        OpenAICompatibleProvider,
    )

    cfg = _config_with_framework_diagram(
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        api_key="k",
        model="dall-e-3",
    )
    providers = build_framework_diagram_providers(
        config=cfg.experiment.framework_diagram,
        figure_config=cfg.experiment.figure_agent,
        output_dir=tmp_path,
        llm=None,
    )
    assert len(providers) == 1
    assert isinstance(providers[0], OpenAICompatibleProvider)


def test_provider_construction_explicit_grsai(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        GrsaiGPTImagesProvider,
        build_framework_diagram_providers,
    )

    cfg = _config_with_framework_diagram(
        provider="grsai_gpt_images", grsai_api_key="g-key"
    )
    providers = build_framework_diagram_providers(
        config=cfg.experiment.framework_diagram,
        figure_config=cfg.experiment.figure_agent,
        output_dir=tmp_path,
        llm=None,
    )
    assert len(providers) == 1
    assert isinstance(providers[0], GrsaiGPTImagesProvider)


def test_provider_construction_explicit_matplotlib_returns_empty(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        build_framework_diagram_providers,
    )

    cfg = _config_with_framework_diagram(provider="matplotlib")
    providers = build_framework_diagram_providers(
        config=cfg.experiment.framework_diagram,
        figure_config=cfg.experiment.figure_agent,
        output_dir=tmp_path,
        llm=None,
    )
    assert providers == []


def test_provider_construction_auto_with_no_creds_returns_empty(tmp_path: Path) -> None:
    """provider='auto' with no API keys → empty list (matplotlib only)."""
    from researchclaw.agents.figure_agent.framework_diagram import (
        build_framework_diagram_providers,
    )

    # Ensure no env vars leak in
    with mock.patch.dict(
        "os.environ",
        {k: "" for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")},
        clear=False,
    ):
        cfg = _config_with_framework_diagram(provider="auto")
        # Force-disable Gemini to make the test deterministic
        from dataclasses import replace

        fig_cfg = replace(
            cfg.experiment.figure_agent,
            nano_banana_enabled=False,
            gemini_api_key="",
        )
        cfg = replace(cfg, experiment=replace(cfg.experiment, figure_agent=fig_cfg))
        providers = build_framework_diagram_providers(
            config=cfg.experiment.framework_diagram,
            figure_config=cfg.experiment.figure_agent,
            output_dir=tmp_path,
            llm=None,
        )
    assert providers == []


# ---------------------------------------------------------------------------
# Matplotlib fallback
# ---------------------------------------------------------------------------


def test_traditional_diagram_produces_real_png(tmp_path: Path) -> None:
    """Fallback writes a real PNG > 1 KB with the PNG signature."""
    from researchclaw.agents.figure_agent.framework_diagram import (
        _render_traditional_framework_diagram,
    )

    out = tmp_path / "charts" / "framework_diagram.png"
    result = _render_traditional_framework_diagram(
        "The encoder processes the data and the training objective evaluates predictions.",
        paper_title="A Test Paper",
        output_path=out,
        dpi=150,
    )

    assert result == out
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 1024
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_traditional_diagram_handles_empty_text(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        _render_traditional_framework_diagram,
    )

    out = tmp_path / "framework_diagram.png"
    _render_traditional_framework_diagram(
        "",
        paper_title="Empty",
        output_path=out,
        dpi=100,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_semantic_skeleton_is_editable_and_topology_locked(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        _render_semantic_skeleton,
    )

    svg = tmp_path / "skeleton.svg"
    layer = tmp_path / "semantic.png"
    nodes, _, _ = _render_semantic_skeleton(
        "EHR record outcome-blind scrub evidence retrieval typed JSON "
        "generator independent cross-family critic commit or abstain",
        paper_title="TRACE-Guard Clinical",
        svg_path=svg,
        layer_png_path=layer,
        dpi=100,
    )
    content = svg.read_text(encoding="utf-8")
    assert "Schema-Bound LLM" in nodes
    assert "Independent Critic" in nodes
    assert "PROPOSED INNOVATION" in content
    assert 'marker-end="url(#arrow)"' in content
    assert "REVISION FEEDBACK" in content
    assert layer.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("introduction", "TRACE-Guard Core"),
        ("method", "Audit Log"),
    ],
)
def test_section_semantic_templates_are_editable_and_detailed(
    tmp_path: Path, section: str, expected: str
) -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        _render_section_semantic_skeleton,
    )

    svg = tmp_path / f"{section}.svg"
    png = tmp_path / f"{section}.png"
    nodes, _, _ = _render_section_semantic_skeleton(
        section=section,
        paper_title="TRACE-Guard Clinical",
        svg_path=svg,
        layer_png_path=png,
    )
    assert expected in nodes
    assert svg.exists() and svg.stat().st_size > 10_000
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_framework_label_fitter_wraps_text_inside_node() -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        _fit_text_to_box,
    )

    label, size = _fit_text_to_box(
        "Parse Failure → Abstain",
        box_width=198,
        box_height=65,
        fontsize=6.5,
    )
    assert "\n" in label
    assert size >= 5.5


def test_framework_collision_validator_rejects_overlap() -> None:
    from researchclaw.agents.figure_agent.framework_diagram import (
        _validate_non_overlapping_regions,
    )

    _validate_non_overlapping_regions([
        ("badge", 0, 0, 20, 20),
        ("title", 24, 0, 50, 20),
    ], min_gap=2)
    with pytest.raises(ValueError, match="badge.*overlaps.*title"):
        _validate_non_overlapping_regions([
            ("badge", 0, 0, 20, 20),
            ("title", 18, 0, 50, 20),
        ], min_gap=2)


# ---------------------------------------------------------------------------
# Orchestrator helper
# ---------------------------------------------------------------------------


def _stub_prompt_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass LLM-driven prompt generation in tests."""
    import researchclaw.agents.figure_agent.framework_diagram as fd_mod

    monkeypatch.setattr(
        fd_mod,
        "_get_real_prompt",
        lambda paper_text, config, llm: (
            "# Framework Diagram Prompt\n"
            "## Image Generation Prompt\n"
            "Architecture diagram body.\n"
            "## Usage Instructions\n"
            "1. Copy above.\n"
        ),
        raising=False,
    )


def test_orchestrator_falls_back_when_all_providers_fail(tmp_path: Path) -> None:
    """Both providers fail → matplotlib fallback writes a real PNG."""
    from researchclaw.agents.figure_agent import framework_diagram as fd

    paper_text = (
        "# A Test Paper\n\n"
        "## Abstract\nWe study encoder + training + evaluation.\n"
    )

    class _FailingProvider:
        name = "fake_failing"

        def generate(self, prompt, *, aspect_ratio, size):
            raise RuntimeError("intentional test failure")

    cfg = _config_with_framework_diagram(
        provider="auto",
        render_mode="direct",
        base_url="https://api.example.com/v1",
        api_key="k",
        model="dall-e-3",
    )

    monkeypatch = pytest.MonkeyPatch()
    try:
        # Bypass LLM prompt generation
        monkeypatch.setattr(
            fd,
            "_generate_framework_diagram_prompt",
            lambda paper_text, config, llm: (
                "# Framework Diagram Prompt\n"
                "## Image Generation Prompt\n"
                "Architecture diagram body.\n"
            ),
            raising=False,
        )
        # Force providers to fail
        monkeypatch.setattr(
            fd,
            "build_framework_diagram_providers",
            lambda **kw: [_FailingProvider(), _FailingProvider()],
        )

        artifacts, png_path = fd.generate_framework_diagram_artifacts(
            paper_text=paper_text,
            config=cfg,
            output_dir=tmp_path,
            llm=None,
        )
    finally:
        monkeypatch.undo()

    assert "framework_diagram_prompt.md" in artifacts
    assert "framework_diagram_image_prompt.txt" in artifacts
    assert "framework_diagram.png" in artifacts
    assert png_path is not None
    assert png_path.exists()
    assert png_path.stat().st_size > 1024
    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    manifest = json.loads((tmp_path / "framework_diagram_generation.json").read_text("utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["original_output"] is None
    assert [item["status"] for item in manifest["generation_attempts"]] == ["failed", "failed"]


def test_orchestrator_uses_first_successful_provider(tmp_path: Path) -> None:
    """First provider raises, second returns bytes → PNG from second."""
    from researchclaw.agents.figure_agent import framework_diagram as fd

    paper_text = "# Test\n"

    class _FailingProvider:
        name = "failing"

        def generate(self, prompt, *, aspect_ratio, size):
            raise RuntimeError("nope")

    class _GoodProvider:
        name = "good"

        def generate(self, prompt, *, aspect_ratio, size):
            return _reviewable_png()

    cfg = _config_with_framework_diagram(
        provider="auto", render_mode="direct"
    )

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            fd,
            "_generate_framework_diagram_prompt",
            lambda paper_text, config, llm: (
                "# Framework Diagram Prompt\n"
                "## Image Generation Prompt\n"
                "X\n"
            ),
            raising=False,
        )
        monkeypatch.setattr(
            fd,
            "build_framework_diagram_providers",
            lambda **kw: [_FailingProvider(), _GoodProvider()],
        )

        artifacts, png_path = fd.generate_framework_diagram_artifacts(
            paper_text=paper_text,
            config=cfg,
            output_dir=tmp_path,
            llm=None,
        )
    finally:
        monkeypatch.undo()

    assert "framework_diagram.png" in artifacts
    assert "framework_diagram_model_original.png" in artifacts
    assert png_path is not None
    assert png_path.read_bytes() == _GOOD_PNG_BYTES
    manifest = json.loads((tmp_path / "framework_diagram_generation.json").read_text("utf-8"))
    prompt_bytes = (tmp_path / manifest["prompt_artifact"]).read_bytes()
    assert prompt_bytes
    assert manifest["prompt_sha256"] == hashlib.sha256(prompt_bytes).hexdigest()
    assert manifest["original_output"] == "framework_diagram_model_original.png"
    assert manifest["original_output_sha256"] == hashlib.sha256(_GOOD_PNG_BYTES).hexdigest()
    assert manifest["image_review"]["status"] == "passed"
    assert manifest["generation_attempts"] == [
        {"provider": "failing", "mode": "prompt", "status": "failed", "error_type": "RuntimeError"},
        {"provider": "good", "mode": "prompt", "status": "succeeded"},
    ]


def test_orchestrator_hybrid_locks_semantics_and_records_candidate(
    tmp_path: Path,
) -> None:
    from researchclaw.agents.figure_agent import framework_diagram as fd

    class _HybridProvider:
        name = "grsai_gpt_images"

        def generate_with_reference(
            self, prompt, reference_image, *, aspect_ratio, size
        ):
            assert reference_image[:8] == b"\x89PNG\r\n\x1a\n"
            assert "immutable layout reference" in prompt
            return reference_image

    cfg = _config_with_framework_diagram(
        provider="grsai_gpt_images",
        render_mode="hybrid",
        visual_influence=0.06,
    )
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            fd,
            "_generate_framework_diagram_prompt",
            lambda paper_text, config, llm: (
                "## Image Generation Prompt\n"
                "EHR record outcome-blind scrub evidence retrieval schema JSON "
                "generator independent critic commit abstain\n"
            ),
            raising=False,
        )
        monkeypatch.setattr(
            fd,
            "build_framework_diagram_providers",
            lambda **kw: [_HybridProvider()],
        )
        artifacts, png_path = fd.generate_framework_diagram_artifacts(
            paper_text="# TRACE-Guard Clinical\n",
            config=cfg,
            output_dir=tmp_path,
            llm=None,
        )
    finally:
        monkeypatch.undo()

    assert png_path is not None and png_path.exists()
    assert "framework_diagram_skeleton.svg" in artifacts
    assert "framework_diagram_semantic_layer.png" in artifacts
    assert "framework_diagram_visual_candidate.png" in artifacts
    manifest = json.loads(
        (tmp_path / "framework_diagram_generation.json").read_text("utf-8")
    )
    assert manifest["render_mode"] == "hybrid"
    assert manifest["schema_version"] == 3
    assert manifest["image_review"]["status"] == "passed"
    assert all(manifest["semantic_lock"].values())
    assert manifest["visual_influence"] == 0.06
    assert manifest["original_output_sha256"] == hashlib.sha256(
        (tmp_path / "framework_diagram_visual_candidate.png").read_bytes()).hexdigest()
    assert manifest["generation_attempts"] == [
        {"provider": "grsai_gpt_images", "mode": "reference", "status": "succeeded"}]


def test_orchestrator_hybrid_refuses_a_degenerate_composite(
    tmp_path: Path,
) -> None:
    """A degenerate hybrid composite ships nothing: file unlinked, no manifest."""
    from researchclaw.agents.figure_agent import framework_diagram as fd

    class _FailingHybridProvider:
        name = "grsai_gpt_images"

        def generate_with_reference(self, *args, **kwargs):
            raise RuntimeError("reference rejected")

    cfg = _config_with_framework_diagram(render_mode="hybrid")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(fd, "build_framework_diagram_providers",
                            lambda **kw: [_FailingHybridProvider()])

        def degenerate_composite(*args, output_path, **kwargs):
            from PIL import Image
            Image.new("RGB", (2000, 1000), "white").save(output_path, format="PNG")

        monkeypatch.setattr(fd, "_compose_hybrid_framework", degenerate_composite)
        artifacts, png_path = fd.generate_framework_diagram_artifacts(
            paper_text="# Test\nInput data model output",
            config=cfg,
            output_dir=tmp_path,
            llm=None,
        )
    finally:
        monkeypatch.undo()

    assert png_path is None
    assert "framework_diagram.png" not in artifacts
    assert not (tmp_path / "framework_diagram.png").exists()
    assert not (tmp_path / "framework_diagram_generation.json").exists()


def test_orchestrator_hybrid_survives_provider_failure(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent import framework_diagram as fd

    class _FailingHybridProvider:
        name = "grsai_gpt_images"

        def generate_with_reference(self, *args, **kwargs):
            raise RuntimeError("gateway rejected reference")

    cfg = _config_with_framework_diagram(render_mode="hybrid")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            fd, "build_framework_diagram_providers",
            lambda **kw: [_FailingHybridProvider()],
        )
        artifacts, png_path = fd.generate_framework_diagram_artifacts(
            paper_text="# Test\nInput data model output",
            config=cfg,
            output_dir=tmp_path,
            llm=None,
        )
    finally:
        monkeypatch.undo()
    assert png_path is not None and png_path.exists()
    assert "framework_diagram_visual_candidate.png" not in artifacts
    manifest = json.loads(
        (tmp_path / "framework_diagram_generation.json").read_text("utf-8")
    )
    assert manifest["provider"] is None


def test_orchestrator_hybrid_retries_text_free_candidate(
    tmp_path: Path,
) -> None:
    from researchclaw.agents.figure_agent import framework_diagram as fd

    class _ReferenceRejectingProvider:
        name = "grsai_gpt_images"

        def generate_with_reference(self, *args, **kwargs):
            raise RuntimeError("data URL unsupported")

        def generate(self, prompt, *, aspect_ratio, size):
            assert "text-free" in prompt
            # Return the valid semantic image supplied by the test renderer.
            return self.valid_png

    provider = _ReferenceRejectingProvider()
    cfg = _config_with_framework_diagram(render_mode="hybrid")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            fd,
            "build_framework_diagram_providers",
            lambda **kw: [provider],
        )
        original = fd._render_semantic_skeleton

        def capture_renderer(*args, **kwargs):
            result = original(*args, **kwargs)
            provider.valid_png = result[2].read_bytes()
            return result

        monkeypatch.setattr(fd, "_render_semantic_skeleton", capture_renderer)
        artifacts, png_path = fd.generate_framework_diagram_artifacts(
            paper_text="# Test\nInput data model output",
            config=cfg,
            output_dir=tmp_path,
            llm=None,
        )
    finally:
        monkeypatch.undo()
    assert png_path is not None and png_path.exists()
    assert "framework_diagram_visual_candidate.png" in artifacts


def test_generation_evidence_verifier_rejects_artifact_and_ledger_tampering(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent import framework_diagram as fd

    class _Provider:
        name = "audited"

        def generate(self, prompt, *, aspect_ratio, size):
            return _GOOD_PNG_BYTES

    cfg = _config_with_framework_diagram(render_mode="direct")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(fd, "build_framework_diagram_providers", lambda **kw: [_Provider()])
        fd.generate_framework_diagram_artifacts(
            paper_text="# Audit\nInput model output", config=cfg, output_dir=tmp_path, llm=None)
    finally:
        monkeypatch.undo()
    assert fd.verify_framework_diagram_artifacts(tmp_path)["provider"] == "audited"

    for name in ("framework_diagram_image_prompt.txt", "framework_diagram_model_original.png",
                 "framework_diagram.png"):
        path = tmp_path / name
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        with pytest.raises(fd.FrameworkDiagramVerificationError):
            fd.verify_framework_diagram_artifacts(tmp_path)
        path.write_bytes(original)

    manifest_path = tmp_path / "framework_diagram_generation.json"
    original_manifest = manifest_path.read_text("utf-8")
    manifest = json.loads(original_manifest)
    manifest["generation_attempts"][0]["error_type"] = "Fabricated"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(fd.FrameworkDiagramVerificationError):
        fd.verify_framework_diagram_artifacts(tmp_path)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    manifest = json.loads(original_manifest)
    manifest["original_output"] = "../outside.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(fd.FrameworkDiagramVerificationError):
        fd.verify_framework_diagram_artifacts(tmp_path)


def test_review_rejected_provider_falls_through_to_the_next_provider(
    tmp_path: Path,
) -> None:
    """A returned-but-unpublishable raster stays a succeeded ledger attempt,
    then the orchestrator moves on to the next provider."""
    from researchclaw.agents.figure_agent import framework_diagram as fd
    from researchclaw.agents.figure_agent.model_image_review import review_model_image
    from researchclaw.llm.image_call_ledger import collect_image_call_records

    tiny = _undersized_reviewable_png()
    assert review_model_image(tiny)["issues"] == ["too_small"]

    class _TinyProvider:
        name = "tiny"

        def generate(self, prompt, *, aspect_ratio, size):
            return tiny

    class _GoodProvider:
        name = "good"

        def generate(self, prompt, *, aspect_ratio, size):
            return _GOOD_PNG_BYTES

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
                            lambda **kw: [_TinyProvider(), _GoodProvider()])
        artifacts, png_path = fd.generate_framework_diagram_artifacts(
            paper_text="# Test\n", config=cfg, output_dir=tmp_path, llm=None)
    finally:
        monkeypatch.undo()

    assert png_path is not None and png_path.read_bytes() == _GOOD_PNG_BYTES
    assert "framework_diagram_model_original.png" in artifacts
    manifest = json.loads((tmp_path / "framework_diagram_generation.json").read_text("utf-8"))
    assert manifest["provider"] == "good"
    assert manifest["image_review"]["status"] == "passed"
    assert [(item["provider"], item["status"])
            for item in manifest["generation_attempts"]] == [
        ("tiny", "succeeded"), ("good", "succeeded")]
    # The run-level accumulator may already hold records from earlier tests
    # in the same process; this run's two attempts are the tail.
    records = collect_image_call_records()[-2:]
    assert [(record["provider"], record["status"]) for record in records] == [
        ("tiny", "succeeded"), ("good", "succeeded")]
    assert records[0]["image_bytes"] == len(tiny)


def test_all_provider_images_failing_review_fall_back_to_matplotlib(
    tmp_path: Path,
) -> None:
    from researchclaw.agents.figure_agent import framework_diagram as fd

    class _TinyProvider:
        name = "tiny"

        def generate(self, prompt, *, aspect_ratio, size):
            return _undersized_reviewable_png()

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
                            lambda **kw: [_TinyProvider(), _TinyProvider()])
        fd.generate_framework_diagram_artifacts(
            paper_text="# Test\n", config=cfg, output_dir=tmp_path, llm=None)
    finally:
        monkeypatch.undo()

    manifest = json.loads((tmp_path / "framework_diagram_generation.json").read_text("utf-8"))
    assert manifest["provider"] == "matplotlib"
    assert manifest["original_output"] is None
    assert manifest["original_output_sha256"] is None
    assert manifest["image_review"]["status"] == "passed"
    assert [item["status"] for item in manifest["generation_attempts"]] == [
        "succeeded", "succeeded"]
    assert fd.verify_framework_diagram_artifacts(tmp_path)["provider"] == "matplotlib"


def test_fallback_refuses_a_degenerate_render(tmp_path: Path) -> None:
    """A degenerate deterministic render ships nothing: no file at the
    fixed-path markdown reference, no manifest, ``None`` returned."""
    from researchclaw.agents.figure_agent import framework_diagram as fd

    class _StaleProvider:
        name = "stale"

        def generate(self, prompt, *, aspect_ratio, size):
            return _undersized_reviewable_png()

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
                            lambda **kw: [_StaleProvider()])

        def degenerate_render(*args, output_path, **kwargs):
            from PIL import Image
            Image.new("RGB", (2000, 1000), "white").save(output_path, format="PNG")

        monkeypatch.setattr(fd, "_render_traditional_framework_diagram",
                            degenerate_render)
        artifacts, png_path = fd.generate_framework_diagram_artifacts(
            paper_text="# Test\n", config=cfg, output_dir=tmp_path, llm=None)
    finally:
        monkeypatch.undo()

    assert png_path is None
    assert "framework_diagram.png" not in artifacts
    assert not (tmp_path / "framework_diagram.png").exists()
    assert not (tmp_path / "framework_diagram_generation.json").exists()


def test_verifier_rejects_review_tampering_and_unreviewable_embeddings(
    tmp_path: Path,
) -> None:
    from researchclaw.agents.figure_agent import framework_diagram as fd
    from researchclaw.agents.figure_agent.model_image_review import review_model_image

    class _Provider:
        name = "audited"

        def generate(self, prompt, *, aspect_ratio, size):
            return _GOOD_PNG_BYTES

    cfg = _config_with_framework_diagram(render_mode="direct")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(fd, "build_framework_diagram_providers",
                            lambda **kw: [_Provider()])
        fd.generate_framework_diagram_artifacts(
            paper_text="# Audit\nInput model output", config=cfg,
            output_dir=tmp_path, llm=None)
    finally:
        monkeypatch.undo()
    manifest_path = tmp_path / "framework_diagram_generation.json"
    original_manifest = manifest_path.read_text("utf-8")

    # Inflated measurement: the recorded review must equal the re-derivation.
    manifest = json.loads(original_manifest)
    manifest["image_review"]["width"] = 2048
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(fd.FrameworkDiagramVerificationError, match="review differs"):
        fd.verify_framework_diagram_artifacts(tmp_path)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    # Schema 3 requires the review block; removing it fails the schema gate.
    manifest = json.loads(original_manifest)
    del manifest["image_review"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(fd.FrameworkDiagramVerificationError, match="schema"):
        fd.verify_framework_diagram_artifacts(tmp_path)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    # A degenerate raster with an honestly recorded failed review is still
    # unpublishable: the second gate rejects the embedding itself.
    import io

    from PIL import Image
    blank = io.BytesIO()
    Image.new("RGB", (2000, 1000), "white").save(blank, format="PNG")
    blank_bytes = blank.getvalue()
    (tmp_path / "framework_diagram.png").write_bytes(blank_bytes)
    manifest = json.loads(original_manifest)
    manifest["provider"] = "matplotlib"
    manifest["original_output"] = None
    manifest["original_output_sha256"] = None
    manifest["image_sha256"] = hashlib.sha256(blank_bytes).hexdigest()
    manifest["image_review"] = review_model_image(blank_bytes)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(fd.FrameworkDiagramVerificationError,
                       match="failed the raster publication review"):
        fd.verify_framework_diagram_artifacts(tmp_path)


def test_orchestrator_disabled_returns_empty(tmp_path: Path) -> None:
    from researchclaw.agents.figure_agent import framework_diagram as fd

    cfg = _config_with_framework_diagram(enabled=False)
    artifacts, png_path = fd.generate_framework_diagram_artifacts(
        paper_text="# x\n",
        config=cfg,
        output_dir=tmp_path,
        llm=None,
    )
    assert artifacts == []
    assert png_path is None


# ---------------------------------------------------------------------------
# Config compatibility (regression)
# ---------------------------------------------------------------------------


def test_legacy_yaml_without_framework_diagram_loads() -> None:
    """Old YAML (no framework_diagram block) loads with sensible defaults."""
    cfg = RCConfig.from_dict(_minimal_yaml())
    assert cfg.experiment.framework_diagram.enabled is True
    assert cfg.experiment.framework_diagram.provider == "auto"
    assert cfg.experiment.framework_diagram.model == "dall-e-3"
    assert cfg.experiment.framework_diagram.grsai_model == "gpt-image-2"
    assert cfg.experiment.framework_diagram.render_mode == "hybrid"
    # Existing figure_agent fields preserved
    assert cfg.experiment.figure_agent.nano_banana_enabled is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "provider",
            "grsai_typo",
            "Invalid experiment.framework_diagram.provider",
        ),
        (
            "grsai_api_style",
            "legacy",
            "Invalid experiment.framework_diagram.grsai_api_style",
        ),
        (
            "render_mode",
            "generative_only",
            "Invalid experiment.framework_diagram.render_mode",
        ),
        (
            "professional_style",
            "cinematic",
            "Invalid experiment.framework_diagram.professional_style",
        ),
    ],
)
def test_framework_diagram_config_rejects_unknown_branch_values(
    field: str, value: str, message: str
) -> None:
    from researchclaw.config import validate_config

    data = _minimal_yaml()
    data["experiment"] = {
        "mode": "sandbox",
        "framework_diagram": {field: value},
    }
    result = validate_config(data, check_paths=False)
    assert any(message in error for error in result.errors)


def test_framework_diagram_config_accepts_grsai_branch() -> None:
    from researchclaw.config import validate_config

    data = _minimal_yaml()
    data["experiment"] = {
        "mode": "sandbox",
        "framework_diagram": {
            "provider": "grsai_gpt_images",
            "grsai_api_style": "openai",
        },
    }
    result = validate_config(data, check_paths=False)
    assert not any("framework_diagram" in error for error in result.errors)


# ---------------------------------------------------------------------------
# Paper placeholder path is unchanged
# ---------------------------------------------------------------------------


def test_paper_placeholder_path_unchanged() -> None:
    """Regression: paper_draft.md still references charts/framework_diagram.png."""
    src = (
        Path(__file__).parents[1]
        / "researchclaw"
        / "pipeline"
        / "stage_impls"
        / "_paper_writing.py"
    ).read_text(encoding="utf-8")
    assert "![Framework Overview](charts/framework_diagram.png)" in src
