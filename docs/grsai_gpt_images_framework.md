# GRSAI GPT-Images framework diagrams

AutoResearchClaw can generate the Stage 22 methodology framework diagram with
GRSAI GPT-Images. This complements the editable, deterministic Matplotlib/SVG
workflow: image generation is useful for rapid visual composition, while the
fallback guarantees that publication export does not depend on an external API.

## Configuration

Set the API key in the environment:

```bash
export GRSAI_API_KEY="your-key"
```

Then configure the experiment:

```yaml
experiment:
  framework_diagram:
    enabled: true
    provider: grsai_gpt_images
    grsai_base_url: https://grsaiapi.com
    grsai_api_key_env: GRSAI_API_KEY
    grsai_model: gpt-image-2
    grsai_api_style: openai
    grsai_quality: auto  # use high with gpt-image-2-vip
    render_mode: hybrid
    professional_style: strict_academic
    visual_influence: 0.06
    size: 1536x1024
    aspect_ratio: "16:9"
    timeout_sec: 180
```

`grsai_api_style: openai` uses `POST /v1/images/generations`. The alternative
`unified` style uses `POST /v1/api/generate`. Do not mix fields from the two
request schemas. A base URL ending in `/v1` is accepted and normalized.

## Generation flow

1. Stage 22 extracts the paper method and deterministically selects semantic
   nodes.
2. It writes an editable `framework_diagram_skeleton.svg` and a transparent
   semantic PNG. These assets lock all labels, nodes, arrows, and feedback
   edges.
3. GRSAI receives the skeleton as an image-to-image reference and may suggest
   only restrained spacing, flat fills, and non-semantic polish.
4. The candidate is signature-checked, whitened to the configured influence
   (maximum 0.15), and placed below the authoritative semantic layer.
5. If the API or reference format fails, the pipeline publishes the pure
   deterministic skeleton rather than losing the figure.
6. The manifest records hashes and explicit semantic locks, but never the key.

`strict_academic` deliberately prohibits gradients, shadows, glow, 3D,
cinematic lighting, cartoons, decorative illustration, and marketing styling.
Hybrid is the default render mode when the field is omitted. Set
`render_mode: direct` only when a deliberately generative bitmap is required.

Use `provider: auto` to try configured GRSAI, OpenAI-compatible, and Gemini
providers in order. Use `provider: matplotlib` for a fully local run.

The raster result can be manually refined, but for fine control over borders,
font sizes, and arrows, retain or recreate the final artwork as SVG/TikZ rather
than repeatedly editing pixels.
