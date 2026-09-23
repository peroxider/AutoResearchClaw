"""Counterexamples for the raster publication gate on model image output."""
import io
import json

import pytest

from researchclaw.agents.figure_agent.model_image_review import review_model_image
from tests.test_framework_diagram import _PNG_BYTES


def _draw(width: int = 1024, height: int = 640, colored: bool = True) -> bytes:
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    if colored:
        # Antialiased-looking diagram-like content: enough distinct colors to
        # clear the degenerate-content floor at the 64x64 downsample.
        for index in range(80):
            shade = index * 3
            draw.rectangle([index * 12 % (width - 40), index * 7 % (height - 40),
                            index * 12 % (width - 40) + 30, index * 7 % (height - 40) + 20],
                           fill=(shade % 256, (shade * 7) % 256, (shade * 13) % 256))
    return_png = io.BytesIO()
    image.save(return_png, format="PNG")
    return return_png.getvalue()


def test_diagram_like_image_passes_with_measured_facts() -> None:
    review = review_model_image(_draw())
    assert review["status"] == "passed" and review["issues"] == []
    assert (review["width"], review["height"]) == (1024, 640)
    assert review["distinct_colors"] >= 64


def test_just_below_the_resolution_floor_is_rejected_with_measurements() -> None:
    review = review_model_image(_draw(width=1023))
    assert review["status"] == "failed" and review["issues"] == ["too_small"]
    assert (review["width"], review["height"]) == (1023, 640)


def test_blank_large_canvas_is_rejected_as_degenerate() -> None:
    review = review_model_image(_draw(colored=False, width=2000, height=1000))
    assert review["status"] == "failed"
    assert review["issues"] == ["degenerate_content"]
    assert (review["width"], review["height"]) == (2000, 1000)


def test_broken_png_body_is_not_decodable() -> None:
    review = review_model_image(_PNG_BYTES)
    assert review["status"] == "failed" and review["issues"] == ["not_decodable"]
    assert review["width"] is None and review["height"] is None
    assert review["distinct_colors"] is None


def test_jpeg_body_is_mis_declared_not_png() -> None:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 640), "white").save(buffer, format="JPEG")
    review = review_model_image(buffer.getvalue())
    assert review["status"] == "failed" and review["issues"] == ["not_png"]
    assert review["width"] is None


@pytest.mark.parametrize("payload", [None, b"", b"GIF89a-not-a-png-at-all"])
def test_absent_or_foreign_bodies_fail_closed(payload) -> None:
    review = review_model_image(payload)
    assert review["status"] == "failed" and review["issues"] == ["not_png"]


def test_review_is_a_pure_function_of_the_bytes() -> None:
    payload = _draw()
    assert review_model_image(payload) == review_model_image(payload)
    assert json.loads(json.dumps(review_model_image(payload))) == review_model_image(payload)
