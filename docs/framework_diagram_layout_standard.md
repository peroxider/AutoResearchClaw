# Framework Diagram Layout Standard

This standard applies to Introduction-level overview diagrams and Method-level
algorithm diagrams, including deterministic, generated, and hybrid rendering.

## Non-negotiable layout constraints

1. **No foreground overlap.** Module badges, module titles, innovation tags,
   nodes, connectors, and annotations must occupy disjoint reserved regions.
   Header badges and titles remain in the header band; innovation tags use a
   separate band below it.
2. **Text must remain inside its node.** Labels are wrapped according to the
   node's usable width. If the resulting lines exceed the usable height, the
   renderer reduces the font size only to the documented minimum. A label that
   still does not fit is a rendering error, not a reason to clip or overflow.
3. **Explicit line breaks are semantic.** Authors may insert line breaks at
   meaningful boundaries such as `Parse Failure` / `Abstain`; the fitter must
   preserve them.
4. **Connector clearance.** Arrows terminate at node edges and must not cross
   text, badges, or headings. Cross-module connectors use boundary anchors and
   orthogonal routing where practical.
5. **Validation before export.** Registered sibling regions are checked for
   collision before SVG and PNG export. Any collision fails generation.
6. **Canvas-safe figure titles.** Diagram headings use the manuscript's short
   title (the primary clause before a colon), not the full article title. They
   must remain inside the horizontal canvas at publication scale.

## Typography and spacing

- Minimum fitted node-label size: 5.5 pt at the 1536 x 900 design canvas.
- Horizontal text padding: at least 14 design pixels on each side.
- Vertical text padding: at least 10 design pixels on each side.
- Minimum gap between registered foreground regions: 2 design pixels; larger
  editorial spacing is preferred.
- Each module header uses three independent zones: badge, title, and optional
  annotation/innovation tag.

## Required review

Every changed framework template must pass automated fitting and collision
tests and receive a visual review of the exported PNG at publication scale.
The editable SVG remains the source for manual fine adjustments.
