"""Framework diagram generation — fills the Stage 22 placeholder PNG.

Today the paper markdown references ``charts/framework_diagram.png`` (see
``_paper_writing.py:2198``) but Stage 22 only writes a text prompt for humans
to paste into DALL-E. This module wires the actual generation:

- :class:`ImageGenProvider` — abstract interface (return image bytes).
- :class:`OpenAICompatibleProvider` — POST ``/images/generations`` against any
  OpenAI-compatible endpoint (DALL-E, Stability, OpenRouter image routes,
  local SD servers, …). Stdlib ``urllib`` only.
- :class:`GeminiProvider` — wraps :class:`NanoBananaAgent` (Google Gemini
  Nano Banana) so we don't duplicate SDK/REST logic.
- :func:`generate_framework_diagram_artifacts` — orchestrator helper used by
  Stage 22 and by tests. Tries providers in configured order, falls back to
  :func:`_render_traditional_framework_diagram` (matplotlib boxes-and-arrows)
  when no provider is available or all providers fail.

The ``framework_diagram_prompt.md`` artifact is still always written so users
can re-render manually if they prefer.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import logging
import os
import re
import textwrap
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image-signature sniffing — reject b64 strings that don't decode to a real
# image so we don't ship a JSON-shaped PNG to disk.
# ---------------------------------------------------------------------------

_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"RIFF", "container"),  # WebP — checked more strictly below
)

_WEBP_MARKER = b"WEBP"


def _looks_like_image(data: bytes) -> bool:
    if not data:
        return False
    for sig, kind in _IMAGE_SIGNATURES:
        if data.startswith(sig):
            if kind == "container":
                return len(data) >= 12 and data[8:12] == _WEBP_MARKER
            return True
    return False


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class ImageGenProvider(ABC):
    """Image-generation backend contract.

    Concrete providers must return raw image bytes (PNG/JPEG/WebP) and never
    write files. The Stage 22 orchestrator owns artifact paths.
    """

    name: ClassVar[str]

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str,
        size: str,
    ) -> bytes:
        """Generate an image. Raise on any failure (transport, auth, empty)."""


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (DALL-E, Stability, OpenRouter image, local SD)
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider(ImageGenProvider):
    """Posts to ``{base_url}/images/generations`` using the OpenAI schema."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: int = 120,
    ) -> None:
        if not base_url:
            raise ValueError("OpenAICompatibleProvider requires base_url")
        if not model:
            raise ValueError("OpenAICompatibleProvider requires model")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_sec = timeout_sec

    def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str,
        size: str,
    ) -> bytes:
        url = f"{self._base_url}/images/generations"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
            "n": 1,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
            body = resp.read().decode("utf-8")

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenAI-compatible endpoint returned non-JSON body: "
                f"{body[:200]}"
            ) from exc

        items = parsed.get("data") or []
        if not items:
            raise RuntimeError(
                "OpenAI-compatible endpoint returned empty 'data' list"
            )
        first = items[0]

        # Preferred: b64_json inline
        b64 = first.get("b64_json")
        if b64:
            image_bytes = base64.b64decode(b64)
        else:
            # Fallback: fetch the URL
            image_url = first.get("url")
            if not image_url:
                raise RuntimeError(
                    "OpenAI-compatible response has neither b64_json nor url"
                )
            with urllib.request.urlopen(
                urllib.request.Request(image_url), timeout=self._timeout_sec
            ) as image_resp:
                image_bytes = image_resp.read()

        if not _looks_like_image(image_bytes):
            raise RuntimeError(
                "OpenAI-compatible response decoded to non-image bytes "
                f"({len(image_bytes)} bytes)"
            )
        return image_bytes


class GrsaiGPTImagesProvider(ImageGenProvider):
    """GRSAI GPT-Images backend for journal-style framework diagrams.

    ``api_style='openai'`` targets the current OpenAI-compatible
    ``/v1/images/generations`` API. ``api_style='unified'`` targets GRSAI's
    ``/v1/api/generate`` API. The provider accepts URL and base64 responses,
    validates the downloaded payload, and never persists credentials.
    """

    name = "grsai_gpt_images"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "gpt-image-2",
        api_style: str = "openai",
        quality: str = "auto",
        timeout_sec: int = 180,
    ) -> None:
        if not base_url:
            raise ValueError("GrsaiGPTImagesProvider requires base_url")
        if not api_key:
            raise ValueError("GrsaiGPTImagesProvider requires api_key")
        if api_style not in {"openai", "unified"}:
            raise ValueError("grsai_api_style must be 'openai' or 'unified'")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._api_style = api_style
        self._quality = quality
        self._timeout_sec = timeout_sec

    def _endpoint(self, suffix: str) -> str:
        root = self._base_url
        if root.endswith("/v1"):
            root = root[:-3]
        return f"{root}/v1/{suffix}"

    @staticmethod
    def _diagram_prompt(prompt: str, aspect_ratio: str) -> str:
        return (
            "Create a publication-quality medical-informatics algorithm "
            f"framework diagram in {aspect_ratio}. Use a white background, "
            "flat two-dimensional geometry, restrained sans-serif typography, "
            "uniform thin strokes, rigid grid alignment, generous whitespace, "
            "and a muted color-blind-safe palette. The visual tone must be "
            "conservative, institutional, technical, serious, and deliberately "
            "formal. No gradients, shadows, glow, 3D, cinematic lighting, "
            "cartoon styling, mascots, decorative illustration, marketing "
            "aesthetics, or invented content.\n\n"
            f"Diagram specification:\n{prompt}"
        )

    def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str,
        size: str,
        reference_image: bytes | None = None,
    ) -> bytes:
        styled_prompt = self._diagram_prompt(prompt, aspect_ratio)
        references: list[str] = []
        if reference_image:
            references.append(
                "data:image/png;base64,"
                + base64.b64encode(reference_image).decode("ascii")
            )
        if self._api_style == "openai":
            url = self._endpoint("images/generations")
            payload = {
                "model": self._model,
                "prompt": styled_prompt,
                "image": references,
                "size": size,
                "response_format": "url",
            }
            if self._quality != "auto":
                payload["quality"] = self._quality
        else:
            url = self._endpoint("api/generate")
            payload = {
                "model": self._model,
                "prompt": styled_prompt,
                "images": references,
                "aspectRatio": size,
                "replyType": "json",
            }
            if self._quality != "auto":
                payload["quality"] = self._quality
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"GRSAI returned non-JSON body: {body[:200]}") from exc

        items = (
            parsed.get("data") if self._api_style == "openai"
            else parsed.get("results")
        ) or []
        if not items:
            detail = parsed.get("message") or parsed.get("error") or "empty result"
            raise RuntimeError(f"GRSAI generation failed: {detail}")
        first = items[0]
        encoded = first.get("b64_json") or first.get("b64")
        if encoded:
            image_bytes = base64.b64decode(encoded)
        else:
            image_url = first.get("url")
            if not image_url:
                raise RuntimeError("GRSAI result has neither image URL nor base64 data")
            with urllib.request.urlopen(
                urllib.request.Request(image_url), timeout=self._timeout_sec
            ) as image_resp:
                image_bytes = image_resp.read()
        if not _looks_like_image(image_bytes):
            raise RuntimeError(
                f"GRSAI response decoded to non-image bytes ({len(image_bytes)} bytes)"
            )
        return image_bytes

    def generate_with_reference(
        self,
        prompt: str,
        reference_image: bytes,
        *,
        aspect_ratio: str,
        size: str,
    ) -> bytes:
        """Generate a visual candidate while conditioning on a locked skeleton."""
        return self.generate(
            prompt,
            aspect_ratio=aspect_ratio,
            size=size,
            reference_image=reference_image,
        )


# ---------------------------------------------------------------------------
# Gemini provider (wraps NanoBananaAgent)
# ---------------------------------------------------------------------------


class GeminiProvider(ImageGenProvider):
    """Adapts :class:`NanoBananaAgent` to the :class:`ImageGenProvider` contract.

    ``generate()`` invokes the agent once with a single architecture-diagram
    request, reads the resulting file, returns its bytes, and cleans up the
    intermediate file. The final ``framework_diagram.png`` is written by the
    Stage 22 orchestrator (not by the provider).
    """

    name = "gemini"

    def __init__(
        self,
        *,
        gemini_api_key: str,
        model: str,
        output_dir: str | Path,
        llm: Any = None,
        aspect_ratio: str = "16:9",
        use_sdk: bool | None = None,
        agent: Any = None,
    ) -> None:
        # Lazy import — NanoBananaAgent pulls in google-genai optionally
        from researchclaw.agents.figure_agent.nano_banana import (
            NanoBananaAgent,
        )

        self._output_dir = Path(output_dir)
        if agent is not None:
            self._agent = agent
        else:
            self._agent = NanoBananaAgent(
                llm,
                gemini_api_key=gemini_api_key or None,
                model=model,
                output_dir=self._output_dir,
                aspect_ratio=aspect_ratio,
                use_sdk=use_sdk,
            )

    def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str,
        size: str,  # ignored — Gemini uses aspect_ratio
    ) -> bytes:
        # NanoBananaAgent uses uuid-like figure_id; pick something stable and
        # unique enough to avoid collisions across retries.
        import uuid

        figure_id = f"framework_diagram_{uuid.uuid4().hex[:8]}"

        result = self._agent.execute({
            "image_figures": [{
                "figure_id": figure_id,
                "figure_type": "architecture_diagram",
                "section": "Method",
                "description": prompt,
            }],
            "topic": "",
            "output_dir": str(self._output_dir),
        })

        generated = (result.data or {}).get("generated") or []
        success_item = next(
            (g for g in generated if g.get("success")), None
        )
        if success_item is None:
            error_msg = (
                generated[0].get("error", "unknown")
                if generated
                else "no items returned"
            )
            raise RuntimeError(f"Gemini generation failed: {error_msg}")

        output_path = (
            success_item.get("output_path")
            or success_item.get("path")
        )
        if not output_path:
            raise RuntimeError(
                "Gemini returned success=True but no output_path"
            )
        p = Path(output_path)
        try:
            image_bytes = p.read_bytes()
        finally:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

        if not _looks_like_image(image_bytes):
            raise RuntimeError(
                f"Gemini returned non-image bytes ({len(image_bytes)} B)"
            )
        return image_bytes


# ---------------------------------------------------------------------------
# Prompt-text extraction
# ---------------------------------------------------------------------------


_PROMPT_HEADING = "## Image Generation Prompt"
_USAGE_HEADING = "## Usage Instructions"


def extract_framework_prompt_text(prompt_markdown: str) -> str:
    """Strip the markdown scaffold from ``_generate_framework_diagram_prompt``.

    The current prompt helper returns a full document with a heading, the
    image-gen prompt body, and a "Usage Instructions" tail. Image-gen APIs
    need only the body — feeding them the full markdown wastes tokens and
    confuses some providers.
    """
    if not prompt_markdown:
        return ""
    start = prompt_markdown.find(_PROMPT_HEADING)
    if start < 0:
        return prompt_markdown.strip()
    body_start = start + len(_PROMPT_HEADING)
    usage_idx = prompt_markdown.find(_USAGE_HEADING, body_start)
    body_end = usage_idx if usage_idx > body_start else len(prompt_markdown)
    return prompt_markdown[body_start:body_end].strip()


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def _resolve_openai_credentials(
    api_key: str, api_key_env: str
) -> str:
    if api_key:
        return api_key
    if api_key_env:
        return os.environ.get(api_key_env, "")
    return ""


def _resolve_gemini_credentials(figure_config: Any) -> str:
    return (
        getattr(figure_config, "gemini_api_key", "")
        or os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("GOOGLE_API_KEY", "")
    )


def build_framework_diagram_providers(
    *,
    config: Any,
    figure_config: Any,
    output_dir: Path,
    llm: Any,
) -> list[ImageGenProvider]:
    """Build the provider list in deterministic order.

    Provider selection:
      - ``"grsai_gpt_images"``:  GRSAI GPT-Images only.
      - ``"openai_compatible"``: only OpenAI-compatible.
      - ``"gemini"``:           only Gemini.
      - ``"matplotlib"``:       [] (matplotlib is the fallback renderer).
      - ``"auto"`` (default):   GRSAI, OpenAI-compatible, then Gemini.

    Providers lacking credentials are silently skipped.
    """
    providers: list[ImageGenProvider] = []
    mode = (getattr(config, "provider", "auto") or "auto").lower()

    def _try_grsai() -> None:
        api_key = _resolve_openai_credentials(
            getattr(config, "grsai_api_key", ""),
            getattr(config, "grsai_api_key_env", "GRSAI_API_KEY"),
        )
        if not api_key:
            return
        try:
            providers.append(GrsaiGPTImagesProvider(
                base_url=getattr(config, "grsai_base_url", "https://grsaiapi.com"),
                api_key=api_key,
                model=getattr(config, "grsai_model", "gpt-image-2"),
                api_style=getattr(config, "grsai_api_style", "openai"),
                quality=getattr(config, "grsai_quality", "auto"),
                timeout_sec=getattr(config, "timeout_sec", 120),
            ))
        except ValueError as exc:
            logger.warning("framework_diagram: skipping GRSAI provider (%s)", exc)

    def _try_openai() -> None:
        api_key = _resolve_openai_credentials(
            getattr(config, "api_key", ""),
            getattr(config, "api_key_env", "OPENAI_API_KEY"),
        )
        if not api_key:
            return
        try:
            providers.append(
                OpenAICompatibleProvider(
                    base_url=getattr(config, "base_url", ""),
                    api_key=api_key,
                    model=getattr(config, "model", "dall-e-3"),
                    timeout_sec=getattr(config, "timeout_sec", 120),
                )
            )
        except ValueError as exc:
            logger.warning(
                "framework_diagram: skipping openai_compatible provider "
                "(%s)",
                exc,
            )

    def _try_gemini() -> None:
        if not getattr(figure_config, "nano_banana_enabled", True):
            return
        gemini_key = _resolve_gemini_credentials(figure_config)
        if not gemini_key:
            return
        try:
            providers.append(
                GeminiProvider(
                    gemini_api_key=gemini_key,
                    model=getattr(
                        figure_config,
                        "gemini_model",
                        "gemini-2.5-flash-image",
                    ),
                    output_dir=output_dir,
                    llm=llm,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "framework_diagram: skipping gemini provider (%s)", exc
            )

    if mode == "grsai_gpt_images":
        _try_grsai()
    elif mode == "openai_compatible":
        _try_openai()
    elif mode == "gemini":
        _try_gemini()
    elif mode == "matplotlib":
        pass
    elif mode == "auto":
        _try_grsai()
        _try_openai()
        _try_gemini()
    else:
        logger.warning(
            "framework_diagram: unknown provider %r — falling back to "
            "matplotlib only",
            mode,
        )

    return providers


# ---------------------------------------------------------------------------
# Matplotlib fallback renderer
# ---------------------------------------------------------------------------


_COMPONENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"(?:encoder|decoder|transformer|attention|convolution|mlp|gnn|resnet|vit)",
        "Neural Network Module",
    ),
    (
        r"(?:loss|objective|criterion|training|optimization)",
        "Training/Optimization",
    ),
    (
        r"(?:data|dataset|input|preprocessing|augmentation)",
        "Data Pipeline",
    ),
    (
        r"(?:output|prediction|inference|evaluation)",
        "Output/Evaluation",
    ),
)

_DEFAULT_COMPONENTS: tuple[str, ...] = (
    "Input Processing",
    "Core Model",
    "Training Loop",
    "Evaluation",
)


def _extract_components(text: str) -> list[str]:
    if not text:
        return list(_DEFAULT_COMPONENTS)
    text_lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    for pattern, label in _COMPONENT_PATTERNS:
        if re.search(pattern, text_lower) and label not in seen:
            found.append(label)
            seen.add(label)
    return found if found else list(_DEFAULT_COMPONENTS)


def _render_traditional_framework_diagram(
    prompt_or_method_text: str,
    *,
    paper_title: str,
    output_path: str | Path,
    dpi: int,
) -> Path:
    """Render a clean boxes-and-arrows methodology diagram via matplotlib.

    Used as the universal fallback when external image-gen providers are
    unavailable. Always produces a real PNG (no in-memory bytes — writes to
    ``output_path`` directly).
    """
    import matplotlib

    matplotlib.use("Agg")  # safe for headless / sandboxed runs
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    components = _extract_components(prompt_or_method_text)[:6]

    fig_w = max(8.0, 1.6 * len(components) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=dpi)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    title_text = paper_title or "Methodology Overview"
    if len(title_text) > 90:
        title_text = title_text[:87] + "..."
    ax.text(
        5.0,
        5.3,
        title_text,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        5.0,
        4.7,
        "Proposed Framework",
        ha="center",
        va="center",
        fontsize=10,
        color="#666666",
    )

    # Layout boxes left-to-right
    palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]
    n = len(components)
    margin_x = 0.8
    span = 10.0 - 2 * margin_x
    box_w = min(1.5, span / max(n, 1) * 0.85)
    gap = (span - n * box_w) / max(n - 1, 1) if n > 1 else 0
    y_center = 2.6
    box_h = 1.2

    centers: list[float] = []
    for i, label in enumerate(components):
        x_left = margin_x + i * (box_w + gap)
        x_center = x_left + box_w / 2
        centers.append(x_center)
        fill = palette[i % len(palette)]
        bbox = FancyBboxPatch(
            (x_left, y_center - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor=fill,
            alpha=0.92,
            zorder=2,
        )
        ax.add_patch(bbox)
        # Choose readable text color based on luminance
        r, g, b = (
            int(fill[1:3], 16),
            int(fill[3:5], 16),
            int(fill[5:7], 16),
        )
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "white" if luminance < 150 else "#222222"

        # Wrap long labels at ~14 chars/line
        wrapped: list[str] = []
        words = label.split()
        cur = ""
        for w in words:
            if len(cur) + len(w) + 1 > 14 and cur:
                wrapped.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            wrapped.append(cur)

        ax.text(
            x_center,
            y_center,
            "\n".join(wrapped),
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
            zorder=3,
        )

    # Arrows between boxes
    for i in range(n - 1):
        arrow = FancyArrowPatch(
            (centers[i] + box_w / 2 + 0.02, y_center),
            (centers[i + 1] - box_w / 2 - 0.02, y_center),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#444444",
            zorder=1,
        )
        ax.add_patch(arrow)

    fig.savefig(
        out,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(
            f"matplotlib fallback did not produce a non-empty PNG at {out}"
        )
    return out


# ---------------------------------------------------------------------------
# Deterministic semantic skeleton and hybrid compositor
# ---------------------------------------------------------------------------


def _semantic_framework_nodes(text: str) -> list[str]:
    """Return ordered, deterministic method nodes from paper terminology."""
    lowered = text.lower()
    candidates = [
        ("Clinical Record", r"record|ehr|dataset|input"),
        ("Outcome-Blind Scrub", r"outcome.?blind|scrub|leakage"),
        ("Evidence Retrieval", r"retriev|evidence|top.?k"),
        ("Schema-Bound LLM", r"schema|json|generator|large language"),
        ("Independent Critic", r"critic|cross.?family|audit"),
        ("Commit or Abstain", r"commit|abstain|reject"),
    ]
    nodes = [label for label, pattern in candidates if re.search(pattern, lowered)]
    if len(nodes) < 4:
        return [
            "Input Data",
            "Controlled Context",
            "Proposed Model",
            "Independent Check",
            "Validated Output",
        ]
    return nodes


def _wrapped_label(label: str, width: int = 13) -> list[str]:
    lines = textwrap.wrap(
        label,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return lines[:3] or [label]


def _fit_text_to_box(
    label: str,
    *,
    box_width: float,
    box_height: float,
    fontsize: float,
) -> tuple[str, float]:
    """Wrap and, if needed, shrink a label until it fits its rectangle."""
    current_size = fontsize
    while current_size >= 5.5:
        max_chars = max(6, int((box_width - 28.0) / (1.25 * current_size)))
        lines: list[str] = []
        for paragraph in label.splitlines() or [label]:
            lines.extend(textwrap.wrap(
                paragraph,
                width=max_chars,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""])
        if len(lines) * current_size * 2.05 <= box_height - 20.0:
            return "\n".join(lines), current_size
        current_size -= 0.5
    raise ValueError(
        f"Label cannot fit box {box_width}x{box_height}: {label!r}"
    )


def _validate_non_overlapping_regions(
    regions: list[tuple[str, float, float, float, float]],
    *,
    min_gap: float = 6.0,
) -> None:
    """Reject collisions between registered sibling foreground regions."""
    for i, (name_a, x_a, y_a, w_a, h_a) in enumerate(regions):
        for name_b, x_b, y_b, w_b, h_b in regions[i + 1:]:
            separated = (
                x_a + w_a + min_gap <= x_b
                or x_b + w_b + min_gap <= x_a
                or y_a + h_a + min_gap <= y_b
                or y_b + h_b + min_gap <= y_a
            )
            if not separated:
                raise ValueError(
                    f"Framework layout collision: {name_a!r} overlaps "
                    f"{name_b!r}"
                )


def _svg_text_lines(label: str, x: float, y: float) -> str:
    lines = _wrapped_label(label)
    line_height = 20
    start_y = y - (len(lines) - 1) * line_height / 2
    return "".join(
        f'<text x="{x:.1f}" y="{start_y + i * line_height:.1f}" '
        f'class="node">{html.escape(line)}</text>'
        for i, line in enumerate(lines)
    )


def _render_semantic_skeleton(
    method_text: str,
    *,
    paper_title: str,
    svg_path: Path,
    layer_png_path: Path,
    dpi: int,
    semantic_nodes: list[str] | None = None,
    innovation_node: str = "Schema-Bound LLM",
    subtitle: str = "Deterministic semantic architecture",
) -> tuple[list[str], Path, Path]:
    """Write the authoritative editable SVG and transparent semantic PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    nodes = semantic_nodes or _semantic_framework_nodes(method_text)
    width, height = 1536, 720
    margin, gap = 55, 28
    box_w = (width - 2 * margin - gap * (len(nodes) - 1)) / len(nodes)
    box_h, y = 156, 290
    palette = ["#DCE8F5", "#DCEFED", "#E3EEE0", "#E9E3F1", "#F3E9D2", "#E8ECEF"]
    strokes = ["#3D6D99", "#2B7A78", "#4F7D4A", "#70558F", "#987126", "#53636F"]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs><marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"8\" refX=\"9\" refY=\"4\" orient=\"auto\"><path d=\"M0,0 L10,4 L0,8 z\" fill=\"#263746\"/></marker></defs>",
        "<style>.title{font:600 34px Arial,sans-serif;fill:#172B3A;text-anchor:middle}.subtitle{font:500 18px Arial,sans-serif;fill:#53636F;text-anchor:middle}.node{font:600 16px Arial,sans-serif;fill:#172B3A;text-anchor:middle;dominant-baseline:middle}.tag{font:700 14px Arial,sans-serif;fill:#6A4C00;text-anchor:middle}</style>",
        f'<text x="{width/2}" y="64" class="title">{html.escape((paper_title or "Methodology Framework")[:78])}</text>',
        f'<text x="{width/2}" y="102" class="subtitle">{html.escape(subtitle)}</text>',
    ]
    centers: list[float] = []
    for i, label in enumerate(nodes):
        x = margin + i * (box_w + gap)
        center = x + box_w / 2
        centers.append(center)
        svg.append(
            f'<rect x="{x:.1f}" y="{y}" width="{box_w:.1f}" height="{box_h}" rx="8" fill="{palette[i]}" stroke="{strokes[i]}" stroke-width="2"/>'
        )
        svg.append(_svg_text_lines(label, center, y + box_h / 2))
        if label == innovation_node:
            svg.append(
                f'<rect x="{x + 18:.1f}" y="{y - 38}" width="{box_w - 36:.1f}" height="28" rx="4" fill="#F5E6B3" stroke="#987126"/>'
            )
            svg.append(
                f'<text x="{center:.1f}" y="{y - 23}" class="tag">PROPOSED INNOVATION</text>'
            )
    for i in range(len(centers) - 1):
        svg.append(
            f'<line x1="{centers[i] + box_w/2 + 4:.1f}" y1="{y + box_h/2}" x2="{centers[i+1] - box_w/2 - 8:.1f}" y2="{y + box_h/2}" stroke="#263746" stroke-width="3" marker-end="url(#arrow)"/>'
        )
    if "Independent Critic" in nodes and "Schema-Bound LLM" in nodes:
        src = centers[nodes.index("Independent Critic")]
        dst = centers[nodes.index("Schema-Bound LLM")]
        svg.append(
            f'<path d="M {src:.1f},{y + box_h + 6} C {src:.1f},585 {dst:.1f},585 {dst:.1f},{y + box_h + 6}" fill="none" stroke="#70558F" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#arrow)"/>'
        )
        svg.append(
            f'<text x="{(src + dst)/2:.1f}" y="580" class="subtitle">REVISION FEEDBACK</text>'
        )
    svg.append("</svg>")
    svg_path.write_text("\n".join(svg), encoding="utf-8")

    # Keep the semantic coordinate system invariant across publication DPI
    # settings. Otherwise point-sized fonts grow relative to a fixed pixel
    # canvas when dpi is raised (e.g. 300), causing label collisions.
    render_dpi = 160
    fig, ax = plt.subplots(
        figsize=(width / render_dpi, height / render_dpi), dpi=render_dpi
    )
    fig.patch.set_alpha(0)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.text(width / 2, 64, (paper_title or "Methodology Framework")[:78], ha="center", va="center", fontsize=17, fontweight="semibold", color="#172B3A")
    ax.text(width / 2, 102, subtitle, ha="center", va="center", fontsize=9, color="#53636F")
    for i, (label, center) in enumerate(zip(nodes, centers)):
        x = center - box_w / 2
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=8", facecolor=palette[i], edgecolor=strokes[i], linewidth=1.5))
        ax.text(center, y + box_h / 2, "\n".join(_wrapped_label(label)), ha="center", va="center", fontsize=7.5, fontweight="semibold", color="#172B3A", linespacing=1.25)
        if label == innovation_node:
            ax.text(center, y - 23, "PROPOSED INNOVATION", ha="center", va="center", fontsize=7, fontweight="bold", color="#6A4C00", bbox={"boxstyle": "round,pad=0.35", "fc": "#F5E6B3", "ec": "#987126", "lw": 0.8})
    for i in range(len(centers) - 1):
        ax.add_patch(FancyArrowPatch((centers[i] + box_w/2 + 4, y + box_h/2), (centers[i+1] - box_w/2 - 8, y + box_h/2), arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color="#263746"))
    if "Independent Critic" in nodes and "Schema-Bound LLM" in nodes:
        src = centers[nodes.index("Independent Critic")]
        dst = centers[nodes.index("Schema-Bound LLM")]
        ax.add_patch(FancyArrowPatch((src, y + box_h + 6), (dst, y + box_h + 6), connectionstyle="arc3,rad=0.35", arrowstyle="-|>", mutation_scale=12, linewidth=1.2, linestyle="--", color="#70558F"))
        ax.text((src + dst)/2, 580, "REVISION FEEDBACK", ha="center", va="center", fontsize=7.5, color="#70558F")
    fig.savefig(
        layer_png_path,
        transparent=True,
        dpi=render_dpi,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)
    return nodes, svg_path, layer_png_path


def _render_section_semantic_skeleton(
    *,
    section: str,
    paper_title: str,
    svg_path: Path,
    layer_png_path: Path,
) -> tuple[list[str], Path, Path]:
    """Render journal-ready Introduction or Method architecture templates.

    Unlike the generic row renderer, these templates deliberately mirror the
    four-module visual grammar used by the GRSAI candidate. All connectors are
    anchored to explicit rectangle-edge coordinates.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    width, height, render_dpi = 1536, 900, 160
    fig, ax = plt.subplots(
        figsize=(width / render_dpi, height / render_dpi), dpi=render_dpi
    )
    fig.patch.set_alpha(0)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    colors = {
        "blue": ("#EDF4FA", "#3D6D99"),
        "teal": ("#ECF7F5", "#2B7A78"),
        "green": ("#F0F6EC", "#4F7D4A"),
        "purple": ("#F4F0F8", "#70558F"),
        "amber": ("#FBF5E7", "#987126"),
        "red": ("#FAEEEE", "#A94A4A"),
    }

    def group(x, y, w, h, title, color, letter):
        fill, stroke = colors[color]
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=10",
            facecolor=fill, edgecolor=stroke, linewidth=1.4,
            linestyle=(0, (5, 4)), alpha=0.72, zorder=1,
        ))
        ax.text(x + 25, y + 37, letter, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                bbox={"boxstyle": "circle,pad=0.35", "fc": stroke,
                      "ec": stroke, "lw": 0.8}, zorder=5)
        ax.text(x + w / 2 + 12, y + 37, title, ha="center", va="center",
                fontsize=7.8, fontweight="bold", color=stroke, zorder=5)

    def box(x, y, w, h, label, color, fontsize=7.6, dashed=False):
        fill, stroke = colors[color]
        fitted_label, fitted_size = _fit_text_to_box(
            label, box_width=w, box_height=h, fontsize=fontsize
        )
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=7",
            facecolor=fill, edgecolor=stroke, linewidth=1.25,
            linestyle="--" if dashed else "-", zorder=3,
        ))
        ax.text(x + w / 2, y + h / 2, fitted_label,
                ha="center", va="center",
                fontsize=fitted_size, fontweight="semibold", color="#172B3A",
                linespacing=1.2, zorder=5)
        return (x, y, w, h)

    def arrow(start, end, color="#263746", curved=0.0, dashed=False):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=12,
            linewidth=1.25, color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={curved}", zorder=4,
            shrinkA=0, shrinkB=0,
        ))

    def elbow_arrow(start, end, bend_x, color="#263746"):
        """Orthogonal cross-module connector with one terminal arrowhead."""
        ax.plot([start[0], bend_x, bend_x],
                [start[1], start[1], end[1]],
                color=color, linewidth=1.25, zorder=4)
        arrow((bend_x, end[1]), end, color=color)

    # A figure heading is not a second abstract. Use the primary title clause
    # so long manuscript titles cannot be clipped at the canvas boundaries.
    title = (paper_title or "TRACE-Guard Clinical").split(":", 1)[0].strip()[:78]
    ax.text(width / 2, 48, title, ha="center", va="center",
            fontsize=17, fontweight="semibold", color="#172B3A")

    if section.lower() in {"introduction", "introduce", "intro"}:
        ax.text(width / 2, 83, "Study-level clinical AI architecture",
                ha="center", va="center", fontsize=9, color="#53636F")
        modules = [
            (25, 145, 330, 650, "INPUT DATA", "blue", "A"),
            (395, 145, 350, 650, "TRACE-GUARD CORE", "teal", "B"),
            (785, 145, 350, 650, "AUDITABLE OUTPUT", "green", "C"),
            (1175, 145, 335, 650, "CLINICAL USE", "purple", "D"),
        ]
        for spec in modules:
            group(*spec)
        left = [
            box(65, 255, 250, 105, "Patient\nCharacteristics", "blue"),
            box(65, 420, 250, 105, "Medications\nand Labs", "blue"),
            box(65, 585, 250, 105, "Clinical\nNotes", "blue"),
        ]
        core = [
            box(435, 235, 270, 105, "Outcome-Blind\nContext", "teal"),
            box(435, 405, 270, 105, "Evidence-Bound\nLLM Generation", "teal"),
            box(435, 575, 270, 105, "Cross-Family\nCritic", "teal"),
        ]
        outputs = [
            box(825, 255, 270, 105, "Risk\nProbability", "green"),
            box(825, 420, 270, 105, "Provenance\nPointers", "green"),
            box(825, 585, 270, 105, "Accept / Revise /\nReject", "green"),
        ]
        use = [
            box(1215, 310, 255, 125, "Commit or\nAbstain", "purple"),
            box(1215, 525, 255, 125, "Clinician\nReview", "purple"),
        ]
        ax.text(570, 215, "PROPOSED INNOVATION", ha="center", va="center",
                fontsize=7.2, fontweight="bold", color="#6A4C00",
                bbox={"boxstyle": "round,pad=0.35", "fc": "#F5E6B3",
                      "ec": "#987126", "lw": 0.9}, zorder=6)
        for stack in (left, core, outputs):
            for a, b in zip(stack, stack[1:]):
                arrow((a[0] + a[2] / 2, a[1] + a[3]),
                      (b[0] + b[2] / 2, b[1]))
        arrow((355, 470), (395, 470))
        arrow((745, 470), (785, 470))
        arrow((1135, 470), (1175, 470))
        arrow((use[0][0] + use[0][2] / 2, use[0][1] + use[0][3]),
              (use[1][0] + use[1][2] / 2, use[1][1]))
        nodes = ["Input Data", "TRACE-Guard Core", "Auditable Output", "Clinical Use"]
    else:
        ax.text(width / 2, 83, "Locked methodology and independent audit flow",
                ha="center", va="center", fontsize=9, color="#53636F")
        modules = [
            (18, 130, 300, 610, "OUTCOME-BLIND INPUT", "blue", "A"),
            (342, 130, 285, 610, "EVIDENCE RETRIEVAL", "teal", "B"),
            (651, 130, 340, 610, "SCHEMA-BOUND LLM", "green", "C"),
            (1015, 130, 503, 610, "CROSS-FAMILY AUDIT", "purple", "D"),
        ]
        for spec in modules:
            group(*spec)
        a1 = box(53, 215, 230, 80, "Patient Record", "blue")
        a2 = box(53, 340, 230, 80, "Label Held Out", "blue")
        a3 = box(53, 465, 230, 80, "Scrub Terms", "blue")
        a4 = box(53, 590, 230, 80, "Leakage Check", "blue")
        b1 = box(377, 285, 215, 95, "Build Clean\nQuery", "teal")
        b2 = box(377, 485, 215, 95, "Retrieve Top-k\nEvidence", "teal")
        c1 = box(690, 225, 262, 82, "Generator", "green")
        c2 = box(690, 355, 262, 132,
                 "Typed JSON\nRisk Score · Features\nProvenance · Confidence",
                 "green", fontsize=6.8)
        c3 = box(690, 540, 262, 82, "Parse Validation", "green")
        c4 = box(722, 650, 198, 65, "Parse Failure\n→ Abstain",
                 "green", fontsize=6.5, dashed=True)
        d1 = box(1105, 215, 320, 75, "Trigger Detection", "purple")
        d2 = box(1105, 335, 320, 75, "Independent Critic", "purple")
        d3 = box(1045, 475, 125, 72, "Accept", "green")
        d4 = box(1205, 475, 125, 72, "Revise", "amber")
        d5 = box(1365, 475, 125, 72, "Reject", "red")
        d6 = box(1045, 610, 180, 70, "Commit Score", "purple")
        d7 = box(1310, 610, 180, 70, "Abstain", "purple")
        ax.text(821, 205, "PROPOSED INNOVATION", ha="center", va="center",
                fontsize=7.2, fontweight="bold", color="#6A4C00",
                bbox={"boxstyle": "round,pad=0.35", "fc": "#F5E6B3",
                      "ec": "#987126", "lw": 0.9}, zorder=6)
        _validate_non_overlapping_regions([
            ("module-C badge", 658, 147, 36, 38),
            ("module-C title", 711, 147, 235, 38),
            ("innovation tag", 706, 191, 230, 28),
            ("generator", c1[0], c1[1], c1[2], c1[3]),
        ], min_gap=2.0)
        audit = box(342, 790, 1176, 72, "AUDIT LOG", "amber", fontsize=9)
        for first, second in ((a1, a2), (a2, a3), (a3, a4),
                              (b1, b2), (c1, c2), (c2, c3),
                              (d1, d2)):
            arrow((first[0] + first[2] / 2, first[1] + first[3]),
                  (second[0] + second[2] / 2, second[1]))
        elbow_arrow((a4[0] + a4[2], a4[1] + a4[3] / 2),
                    (b1[0], b1[1] + b1[3] / 2), bend_x=330)
        elbow_arrow((b2[0] + b2[2], b2[1] + b2[3] / 2),
                    (c1[0], c1[1] + c1[3] / 2), bend_x=639)
        elbow_arrow((c3[0] + c3[2], c3[1] + c3[3] / 2),
                    (d1[0], d1[1] + d1[3] / 2), bend_x=1003)
        arrow((c3[0] + c3[2] / 2, c3[1] + c3[3]),
              (c4[0] + c4[2] / 2, c4[1]))
        for target in (d3, d4, d5):
            arrow((d2[0] + d2[2] / 2, d2[1] + d2[3]),
                  (target[0] + target[2] / 2, target[1]), curved=0.0)
        arrow((d3[0] + d3[2] / 2, d3[1] + d3[3]),
              (d6[0] + d6[2] / 2, d6[1]))
        arrow((d5[0] + d5[2] / 2, d5[1] + d5[3]),
              (d7[0] + d7[2] / 2, d7[1]))
        arrow((d4[0], d4[1] + d4[3] / 2),
              (c1[0] + c1[2], c1[1] + c1[3] / 2),
              color="#70558F", curved=0.28, dashed=True)
        for source in (c4, d4, d6, d7):
            arrow((source[0] + source[2] / 2, source[1] + source[3]),
                  (source[0] + source[2] / 2, audit[1]), color="#987126")
        nodes = ["Outcome-Blind Input", "Evidence Retrieval",
                 "Schema-Bound LLM", "Cross-Family Audit", "Audit Log"]

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg", transparent=True,
                bbox_inches=None, pad_inches=0)
    fig.savefig(layer_png_path, format="png", transparent=True,
                dpi=render_dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return nodes, svg_path, layer_png_path


def _compose_hybrid_framework(
    *, semantic_layer: bytes, candidate: bytes | None, output_path: Path, influence: float
) -> Path:
    """Composite a faint aesthetic candidate under an authoritative layer."""
    from PIL import Image, ImageFilter

    semantic = Image.open(io.BytesIO(semantic_layer)).convert("RGBA")
    canvas = Image.new("RGBA", semantic.size, "white")
    if candidate and influence > 0:
        visual = Image.open(io.BytesIO(candidate)).convert("RGB").resize(semantic.size)
        # Remove all model-generated glyphs and edge topology. Only broad
        # palette/spacing cues may survive beneath the authoritative layer.
        visual = visual.filter(ImageFilter.GaussianBlur(radius=18))
        white = Image.new("RGB", semantic.size, "white")
        faded = Image.blend(white, visual, min(max(influence, 0.0), 0.15))
        canvas = faded.convert("RGBA")
    final = Image.alpha_composite(canvas, semantic)
    # Avoid Pillow's optimize path here: on dense antialiased semantic layers
    # it has produced intermittently corrupted IDAT output on Windows-mounted
    # workspaces. Standard PNG compression is lossless and deterministic.
    final.convert("RGB").save(output_path, format="PNG", compress_level=6)
    return output_path


# ---------------------------------------------------------------------------
# Orchestrator helper (used by Stage 22 and by tests)
# ---------------------------------------------------------------------------


class FrameworkDiagramVerificationError(ValueError):
    pass


def verify_framework_diagram_artifacts(output_dir: Path) -> dict:
    """Verify the portable Stage 22 image-generation evidence chain."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "framework_diagram_generation.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrameworkDiagramVerificationError("Missing or invalid framework diagram manifest") from exc
    common = {"schema_version", "artifact", "render_mode", "provider", "generation_attempts", "model",
              "size", "aspect_ratio", "prompt_sha256", "prompt_artifact", "original_output_sha256",
              "image_sha256"}
    hybrid = {"semantic_source", "semantic_nodes", "visual_candidate", "professional_style",
              "visual_influence", "semantic_lock", "skeleton_sha256"}
    direct = {"original_output"}
    mode = manifest.get("render_mode")
    expected_fields = common | (hybrid if mode == "hybrid" else direct if mode == "direct" else set())
    if set(manifest) != expected_fields or manifest.get("schema_version") != 2:
        raise FrameworkDiagramVerificationError("Invalid framework diagram manifest schema")

    def payload(name, expected_name, digest_field, *, image=False):
        if name != expected_name:
            raise FrameworkDiagramVerificationError("Unexpected framework diagram artifact path")
        path = output_dir / expected_name
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise FrameworkDiagramVerificationError("Missing framework diagram artifact") from exc
        if hashlib.sha256(data).hexdigest() != manifest[digest_field]:
            raise FrameworkDiagramVerificationError("Framework diagram artifact hash differs")
        if image and not _looks_like_image(data):
            raise FrameworkDiagramVerificationError("Framework diagram artifact is not an image")
        return data

    prompt = payload(manifest["prompt_artifact"], "framework_diagram_image_prompt.txt", "prompt_sha256")
    try:
        prompt.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrameworkDiagramVerificationError("Framework diagram prompt is not UTF-8") from exc
    payload(manifest["artifact"], "framework_diagram.png", "image_sha256", image=True)
    attempts = manifest["generation_attempts"]
    if not isinstance(attempts, list) or len(attempts) > 16:
        raise FrameworkDiagramVerificationError("Invalid framework diagram attempt ledger")
    for item in attempts:
        base = {"provider", "mode", "status"}
        if (not isinstance(item, dict) or frozenset(item) not in {frozenset(base), frozenset(base | {"error_type"})}
                or item.get("status") not in {"succeeded", "failed"}
                or not all(isinstance(item.get(key), str) and item[key] for key in base)
                or (item["status"] == "failed") != ("error_type" in item)):
            raise FrameworkDiagramVerificationError("Invalid framework diagram attempt entry")
    if not isinstance(manifest["provider"], (str, type(None))) or not isinstance(manifest["model"], (str, type(None))):
        raise FrameworkDiagramVerificationError("Invalid framework diagram provider identity")
    if mode == "direct":
        original = manifest["original_output"]
        if original is None:
            if manifest["provider"] != "matplotlib" or manifest["original_output_sha256"] is not None:
                raise FrameworkDiagramVerificationError("Direct fallback has inconsistent original output")
        else:
            payload(original, "framework_diagram_model_original.png", "original_output_sha256", image=True)
            if manifest["provider"] in {None, "matplotlib"}:
                raise FrameworkDiagramVerificationError("Model original lacks a model provider")
    else:
        payload(manifest["semantic_source"], "framework_diagram_skeleton.svg", "skeleton_sha256")
        candidate = manifest["visual_candidate"]
        if candidate is None:
            if manifest["provider"] is not None or manifest["original_output_sha256"] is not None:
                raise FrameworkDiagramVerificationError("Hybrid fallback has inconsistent candidate")
        else:
            payload(candidate, "framework_diagram_visual_candidate.png", "original_output_sha256", image=True)
            if manifest["provider"] is None:
                raise FrameworkDiagramVerificationError("Hybrid candidate lacks a provider")
        if manifest["semantic_lock"] != {"labels": True, "nodes": True, "arrows": True,
                                         "module_boundaries": True}:
            raise FrameworkDiagramVerificationError("Hybrid semantic lock is incomplete")
    return manifest


def generate_framework_diagram_artifacts(
    *,
    paper_text: str,
    config: Any,
    output_dir: Path,
    llm: Any = None,
) -> tuple[list[str], Path | None]:
    """Build the framework-diagram artifacts in ``output_dir``.

    Always writes ``framework_diagram_prompt.md`` (so users can re-render
    manually). Tries configured image-gen providers in order. If all fail
    (or no provider is configured), renders the matplotlib fallback. Returns
    the list of artifact paths (relative to ``output_dir``) and the absolute
    path to the final PNG (or ``None`` if matplotlib also failed).
    """
    from researchclaw.pipeline._helpers import (
        _extract_paper_title,
        _generate_framework_diagram_prompt,
    )

    framework_cfg = config.experiment.framework_diagram
    figure_cfg = config.experiment.figure_agent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not framework_cfg.enabled:
        logger.info("framework_diagram: disabled via config — skipping")
        return [], None

    # 1. Generate text prompt (reuses existing LLM-driven helper)
    try:
        prompt_md = _generate_framework_diagram_prompt(
            paper_text, config, llm=llm
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "framework_diagram: prompt generation failed (%s) — using "
            "method-section fallback",
            exc,
        )
        prompt_md = ""

    prompt_path = output_dir / "framework_diagram_prompt.md"
    if prompt_md:
        prompt_path.write_text(prompt_md, encoding="utf-8")
    artifacts: list[str] = []
    if prompt_path.exists():
        artifacts.append(prompt_path.name)

    image_prompt = extract_framework_prompt_text(prompt_md) or paper_text
    image_prompt_path = output_dir / "framework_diagram_image_prompt.txt"
    image_prompt_path.write_text(image_prompt, encoding="utf-8", newline="")
    artifacts.append(image_prompt_path.name)

    # Hybrid mode: the SVG and transparent PNG are the semantic source of
    # truth. Image models may provide only a faint visual candidate beneath
    # this locked layer, so hallucinated labels/arrows cannot enter the paper.
    render_mode = getattr(framework_cfg, "render_mode", "direct")
    if render_mode == "hybrid":
        skeleton_svg = output_dir / "framework_diagram_skeleton.svg"
        semantic_png = output_dir / "framework_diagram_semantic_layer.png"
        if "trace-guard" in paper_text.lower():
            nodes, _, _ = _render_section_semantic_skeleton(
                section="method",
                paper_title=_extract_paper_title(paper_text),
                svg_path=skeleton_svg,
                layer_png_path=semantic_png,
            )
        else:
            nodes, _, _ = _render_semantic_skeleton(
                image_prompt or paper_text,
                paper_title=_extract_paper_title(paper_text),
                svg_path=skeleton_svg,
                layer_png_path=semantic_png,
                dpi=getattr(framework_cfg, "dpi", 300),
            )
        artifacts.extend([skeleton_svg.name, semantic_png.name])

        providers = build_framework_diagram_providers(
            config=framework_cfg,
            figure_config=figure_cfg,
            output_dir=output_dir,
            llm=llm,
        )
        reference_bytes = semantic_png.read_bytes()
        candidate_bytes: bytes | None = None
        candidate_provider: str | None = None
        generation_attempts: list[dict[str, Any]] = []
        enhancement_prompt = (
            "Treat the attached diagram only as an immutable layout reference. "
            "Do not add, remove, rewrite, or relocate nodes, labels, arrows, "
            "or module boundaries. Suggest only restrained spacing, flat fills, "
            "and subtle non-semantic visual polish. Preserve the deliberately "
            "formal clinical-journal appearance. "
            + image_prompt
        )
        for provider in providers:
            try:
                reference_generate = getattr(
                    provider, "generate_with_reference", None
                )
                if reference_generate is not None:
                    try:
                        candidate_bytes = reference_generate(
                            enhancement_prompt,
                            reference_bytes,
                            aspect_ratio=framework_cfg.aspect_ratio,
                            size=framework_cfg.size,
                        )
                        generation_attempts.append({"provider": provider.name, "mode": "reference",
                                                    "status": "succeeded"})
                    except Exception as reference_exc:  # noqa: BLE001
                        logger.warning(
                            "framework_diagram: provider %s rejected the "
                            "skeleton reference (%s); retrying a non-semantic "
                            "visual candidate",
                            provider.name,
                            reference_exc,
                        )
                        candidate_bytes = provider.generate(
                            "Generate only a restrained, abstract, text-free "
                            "clinical-journal background using flat muted "
                            "blue, teal, green, purple, and amber regions on "
                            "white. No words, letters, numbers, arrows, icons, "
                            "gradients, shadows, 3D, decoration, or marketing "
                            "styling.",
                            aspect_ratio=framework_cfg.aspect_ratio,
                            size=framework_cfg.size,
                        )
                        generation_attempts.append({"provider": provider.name, "mode": "reference",
                                                    "status": "failed",
                                                    "error_type": type(reference_exc).__name__})
                        generation_attempts.append({"provider": provider.name, "mode": "text_free",
                                                    "status": "succeeded"})
                else:
                    candidate_bytes = provider.generate(
                        enhancement_prompt,
                        aspect_ratio=framework_cfg.aspect_ratio,
                        size=framework_cfg.size,
                    )
                    generation_attempts.append({"provider": provider.name, "mode": "prompt",
                                                "status": "succeeded"})
                candidate_provider = provider.name
                break
            except Exception as exc:  # noqa: BLE001
                generation_attempts.append({"provider": provider.name, "mode": "provider",
                                            "status": "failed", "error_type": type(exc).__name__})
                logger.warning(
                    "framework_diagram: hybrid candidate provider %s failed (%s)",
                    provider.name,
                    exc,
                )
        candidate_path = output_dir / "framework_diagram_visual_candidate.png"
        if candidate_bytes:
            candidate_path.write_bytes(candidate_bytes)
            artifacts.append(candidate_path.name)

        png_path = output_dir / "framework_diagram.png"
        _compose_hybrid_framework(
            semantic_layer=reference_bytes,
            candidate=candidate_bytes,
            output_path=png_path,
            influence=getattr(framework_cfg, "visual_influence", 0.06),
        )
        generated_via = (
            f"hybrid:{candidate_provider}"
            if candidate_provider
            else "hybrid:deterministic_only"
        )
        artifacts.append(png_path.name)
        manifest_path = output_dir / "framework_diagram_generation.json"
        manifest = {
            "schema_version": 2,
            "artifact": png_path.name,
            "render_mode": "hybrid",
            "semantic_source": skeleton_svg.name,
            "semantic_nodes": nodes,
            "visual_candidate": (
                candidate_path.name if candidate_bytes else None
            ),
            "provider": candidate_provider,
            "generation_attempts": generation_attempts,
            "model": (
                framework_cfg.grsai_model
                if candidate_provider == "grsai_gpt_images"
                else getattr(framework_cfg, "model", None)
            ),
            "professional_style": getattr(
                framework_cfg, "professional_style", "strict_academic"
            ),
            "visual_influence": getattr(
                framework_cfg, "visual_influence", 0.06
            ),
            "semantic_lock": {
                "labels": True,
                "nodes": True,
                "arrows": True,
                "module_boundaries": True,
            },
            "size": framework_cfg.size,
            "aspect_ratio": framework_cfg.aspect_ratio,
            "prompt_sha256": hashlib.sha256(
                image_prompt.encode("utf-8")
            ).hexdigest(),
            "prompt_artifact": image_prompt_path.name,
            "skeleton_sha256": hashlib.sha256(
                skeleton_svg.read_bytes()
            ).hexdigest(),
            "image_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
            "original_output_sha256": (hashlib.sha256(candidate_bytes).hexdigest()
                                        if candidate_bytes else None),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts.append(manifest_path.name)
        verify_framework_diagram_artifacts(output_dir)
        logger.info(
            "framework_diagram: wrote locked hybrid %s via %s",
            png_path.name,
            generated_via,
        )
        return artifacts, png_path

    # 2. Try configured providers
    providers = build_framework_diagram_providers(
        config=framework_cfg,
        figure_config=figure_cfg,
        output_dir=output_dir,
        llm=llm,
    )

    png_path = output_dir / "framework_diagram.png"
    generated_via: str | None = None
    generation_attempts: list[dict[str, Any]] = []
    original_path = output_dir / "framework_diagram_model_original.png"
    for provider in providers:
        try:
            image_bytes = provider.generate(
                image_prompt,
                aspect_ratio=framework_cfg.aspect_ratio,
                size=framework_cfg.size,
            )
        except Exception as exc:  # noqa: BLE001
            generation_attempts.append({"provider": provider.name, "mode": "prompt", "status": "failed",
                                        "error_type": type(exc).__name__})
            logger.warning(
                "framework_diagram: provider %s failed (%s)",
                provider.name,
                exc,
            )
            continue
        if image_bytes:
            original_path.write_bytes(image_bytes)
            png_path.write_bytes(image_bytes)
            generated_via = provider.name
            generation_attempts.append({"provider": provider.name, "mode": "prompt", "status": "succeeded"})
            break

    # 3. Fall back to matplotlib
    if generated_via is None:
        try:
            _render_traditional_framework_diagram(
                image_prompt or paper_text,
                paper_title=_extract_paper_title(paper_text),
                output_path=png_path,
                dpi=getattr(framework_cfg, "dpi", None)
                or getattr(figure_cfg, "dpi", 300),
            )
            generated_via = "matplotlib"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "framework_diagram: matplotlib fallback failed (%s) — "
                "PNG missing; paper placeholder will be broken",
                exc,
            )
            return artifacts, None

    if png_path.exists() and png_path.stat().st_size > 0:
        if original_path.exists():
            artifacts.append(original_path.name)
        artifacts.append(png_path.name)
        manifest_path = output_dir / "framework_diagram_generation.json"
        manifest = {
            "schema_version": 2,
            "artifact": png_path.name,
            "render_mode": "direct",
            "provider": generated_via,
            "generation_attempts": generation_attempts,
            "model": (
                framework_cfg.grsai_model
                if generated_via == "grsai_gpt_images"
                else getattr(framework_cfg, "model", None)
            ),
            "size": framework_cfg.size,
            "aspect_ratio": framework_cfg.aspect_ratio,
            "prompt_sha256": hashlib.sha256(
                image_prompt.encode("utf-8")
            ).hexdigest(),
            "prompt_artifact": image_prompt_path.name,
            "original_output": original_path.name if original_path.exists() else None,
            "original_output_sha256": (hashlib.sha256(original_path.read_bytes()).hexdigest()
                                        if original_path.exists() else None),
            "image_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts.append(manifest_path.name)
        verify_framework_diagram_artifacts(output_dir)
        logger.info(
            "framework_diagram: wrote %s via %s (%d bytes)",
            png_path.name,
            generated_via,
            png_path.stat().st_size,
        )

    return artifacts, png_path


__all__ = [
    "ImageGenProvider",
    "OpenAICompatibleProvider",
    "GrsaiGPTImagesProvider",
    "GeminiProvider",
    "extract_framework_prompt_text",
    "build_framework_diagram_providers",
    "generate_framework_diagram_artifacts",
    "verify_framework_diagram_artifacts",
    "FrameworkDiagramVerificationError",
    "_render_traditional_framework_diagram",
    "_render_semantic_skeleton",
    "_render_section_semantic_skeleton",
    "_compose_hybrid_framework",
    "_fit_text_to_box",
    "_validate_non_overlapping_regions",
]
