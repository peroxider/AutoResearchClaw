"""Raster publication gate for Stage-22 image-model output.

In direct render mode the model's raw bytes became ``framework_diagram.png``
without any inspection: a 64-pixel thumbnail or a blank canvas would ship
straight into the paper. This module is the machine-checkable part of the
diagram publication gate for such rasters — decode integrity (PNG signature,
PIL-decodable), a resolution floor (>= 1024x640 pixels, roughly 150 effective
DPI at a full text-width figure), and non-degenerate content (a downsampled
color-count floor that rejects blank or flat-fill canvases).

This is a legibility and content gate only. It cannot confirm that the model
depicted the requested structure: semantic and aesthetic agreement of a model
image with the declared method remains a human review obligation, and the
hybrid mode keeps the authoritative semantic layer above any model candidate
for exactly that reason.

The review is a pure function of the bytes: the same input always yields the
same recorded measurements, so a frozen manifest's review entry can be
re-derived from the frozen image file and compared byte-for-byte.
"""
from __future__ import annotations

import io

# ~150 effective DPI at a full-width figure; endpoints returning smaller
# thumbnails fail the gate instead of being embedded unreadable.
MIN_WIDTH = 1024
MIN_HEIGHT = 640
# A 64x64 downsample of a real diagram keeps dozens of distinct colors
# (antialiased text over a palette); a blank or flat-fill canvas stays single
# digits. 64 separates them with margin on both sides.
MIN_DISTINCT_COLORS = 64
_DOWNSAMPLE = (64, 64)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def review_model_image(image_bytes: bytes | bytearray | None, *,
                       min_width: int = MIN_WIDTH,
                       min_height: int = MIN_HEIGHT,
                       min_distinct_colors: int = MIN_DISTINCT_COLORS) -> dict:
    """Review raw provider bytes; rejection is a recorded outcome, not an error.

    Returns ``{"status", "issues", "width", "height", "distinct_colors"}``.
    ``width``/``height``/``distinct_colors`` are measured facts, null when the
    image could not be decoded far enough to measure them.
    """
    issues: list[str] = []
    width = height = distinct_colors = None
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        issues.append("not_png")
    elif not bytes(image_bytes[:8]) == _PNG_SIGNATURE:
        # The artifact is declared and stored as .png; a JPEG or WebP body is
        # mis-declared regardless of whether PIL could decode it.
        issues.append("not_png")
    else:
        try:
            from PIL import Image
            image = Image.open(io.BytesIO(bytes(image_bytes)))
            image.load()
            width, height = image.size
            sampled = image.convert("RGB").resize(_DOWNSAMPLE)
            raw = sampled.tobytes()
            distinct_colors = len({(raw[index] << 16) | (raw[index + 1] << 8) | raw[index + 2]
                                   for index in range(0, len(raw), 3)})
        except Exception:  # noqa: BLE001 — any decode failure is a rejection
            issues.append("not_decodable")
            width = height = distinct_colors = None
        else:
            if width < min_width or height < min_height:
                issues.append("too_small")
            if distinct_colors < min_distinct_colors:
                issues.append("degenerate_content")
    return {"status": "failed" if issues else "passed", "issues": issues,
            "width": width, "height": height, "distinct_colors": distinct_colors}
