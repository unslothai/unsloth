// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  borderImageValue,
  imageLayers,
  nineSliceGeometry,
  parseShadowLayers,
  remainingShadow,
  spreadRadius,
} from "../src/lib/cpu-paint-shadow-geometry.ts";

const ROUND_28 = { tl: 28, tr: 28, br: 28, bl: 28 };

test("reads computed and specified shadows, colors with commas included", () => {
  assert.deepEqual(parseShadowLayers("rgba(0, 0, 0, 0.16) 0px 12px 32px -12px, rgba(0, 0, 0, 0.06) 0px 2px 6px -2px"), [
    { color: "rgba(0, 0, 0, 0.16)", x: 0, y: 12, blur: 32, spread: -12, inset: false },
    { color: "rgba(0, 0, 0, 0.06)", x: 0, y: 2, blur: 6, spread: -2, inset: false },
  ]);
  assert.deepEqual(parseShadowLayers("0 4px 20px 4px color(srgb 0.1 0.1 0.1)"), [
    { color: "color(srgb 0.1 0.1 0.1)", x: 0, y: 4, blur: 20, spread: 4, inset: false },
  ]);
  assert.deepEqual(parseShadowLayers("inset 1px 0 0 0 #ccc"), [
    { color: "#ccc", x: 1, y: 0, blur: 0, spread: 0, inset: true },
  ]);
  assert.deepEqual(parseShadowLayers("none"), []);
});

test("anything it cannot measure keeps the CSS shadow", () => {
  assert.equal(parseShadowLayers("0 1rem 2rem black"), null);
  assert.equal(parseShadowLayers("0 4px 20px 4px var(--background)"), null);
  assert.equal(parseShadowLayers("0 2 8px black"), null);
  assert.equal(parseShadowLayers("black"), null);
});

test("rings join the image once a layer is blurred, so they stay above the blur", () => {
  const neon = parseShadowLayers(
    "rgba(230, 210, 0, 0.7) 0px 0px 0px 1px, rgba(252, 238, 10, 0.8) 0px 0px 26px -6px, rgba(0, 0, 0, 0.1) 0px 2px 8px -2px",
  );
  assert.ok(neon);
  assert.equal(imageLayers(neon).length, 3);
  assert.equal(remainingShadow(neon), "0 0 #0000");

  const ringOnly = parseShadowLayers("rgba(0, 0, 0, 0.1) 0px 0px 0px 1px");
  assert.ok(ringOnly);
  assert.deepEqual(imageLayers(ringOnly), []);
  assert.equal(remainingShadow(ringOnly), "0px 0px 0px 1px rgba(0, 0, 0, 0.1)");

  const withInset = parseShadowLayers("inset 0 1px 0 0 #fff, 0 2px 8px -2px black");
  assert.ok(withInset);
  assert.equal(remainingShadow(withInset), "inset 0px 1px 0px 0px #fff");
});

test("corner radii follow the spread as css-backgrounds-3 says", () => {
  assert.equal(spreadRadius(28, -12), 16);
  assert.equal(spreadRadius(10, -12), 0);
  assert.equal(spreadRadius(28, 4), 32);
  assert.equal(spreadRadius(0, 4), 0);
  // r < spread: r + s * (1 + (r/s - 1)^3)
  assert.equal(spreadRadius(2, 4), 2 + 4 * (1 + (0.5 - 1) ** 3));
});

test("slices are whole device pixels and the box edge lands on one", () => {
  const dark = parseShadowLayers("rgba(0, 0, 0, 0.7) 0px 16px 40px -12px");
  assert.ok(dark);
  for (const scale of [1, 1.25, 1.5, 2, 2.5]) {
    const g = nineSliceGeometry(dark, ROUND_28, scale, 200, 120);
    assert.ok(g);
    assert.ok(Number.isInteger(g.outsetPx) && Number.isInteger(g.x.slice) && Number.isInteger(g.y.slice));
    assert.equal(g.outset * scale, g.outsetPx);
    assert.equal(g.x.width * scale, g.x.slice);
    assert.equal(g.y.width * scale, g.y.slice);
    // Blur tail (1.5 x 40) plus the 16px offset fits inside the outset.
    assert.ok(g.outset >= 1.5 * 40 + 16);
    assert.equal(g.imageWidth, 2 * g.x.slice + g.middle);
    assert.equal(g.imageHeight, 2 * g.y.slice + g.middle);
  }
});

test("a positive spread widens the outset; a negative one does not", () => {
  const grow = parseShadowLayers("black 0px 4px 20px 4px");
  const shrink = parseShadowLayers("black 0px 4px 20px -4px");
  assert.ok(grow && shrink);
  const a = nineSliceGeometry(grow, ROUND_28, 1, 400, 400);
  const b = nineSliceGeometry(shrink, ROUND_28, 1, 400, 400);
  assert.ok(a && b);
  assert.equal(a.outset - b.outset, 4);
});

test("corner tiles cover the corner's reach but never force CSS to shrink the slices", () => {
  const dark = parseShadowLayers("rgba(24, 24, 24) 0px 4px 20px 4px");
  assert.ok(dark);
  for (const [minWidth, minHeight] of [
    [200, 48],
    [356, 56],
    [120, 48],
    [400, 400],
  ]) {
    for (const scale of [1, 1.25, 2, 2.5]) {
      const g = nineSliceGeometry(dark, ROUND_28, scale, minWidth, minHeight);
      assert.ok(g);
      // CSS scales every slice down when two widths exceed the box plus both outsets.
      assert.ok(2 * g.x.width <= minWidth + 2 * g.outset, `x ${minWidth} @${scale}`);
      assert.ok(2 * g.y.width <= minHeight + 2 * g.outset, `y ${minHeight} @${scale}`);
    }
  }
  // Room to spare: the tile runs radius + blur tail + offset past the edge.
  const roomy = nineSliceGeometry(dark, ROUND_28, 1, 400, 400);
  assert.ok(roomy);
  assert.equal(roomy.x.width - roomy.outset, 28 + 30 + 4 + 2);
});

test("blur-free shadows need no image", () => {
  const ring = parseShadowLayers("black 0px 0px 0px 1px");
  assert.ok(ring);
  assert.equal(nineSliceGeometry(ring, ROUND_28, 2, 400, 400), null);
  assert.equal(nineSliceGeometry([], ROUND_28, 2, 400, 400), null);
});

test("the border-image value has no fill, so the box keeps its own background", () => {
  const layers = parseShadowLayers("black 0px 2px 8px -2px");
  assert.ok(layers);
  const g = nineSliceGeometry(layers, ROUND_28, 2, 200, 48);
  assert.ok(g);
  const value = borderImageValue("data:image/png;base64,AAAA", g);
  assert.equal(
    value,
    `url("data:image/png;base64,AAAA") ${g.y.slice} ${g.x.slice} / ${g.y.width}px ${g.x.width}px / ${g.outset}px stretch`,
  );
  assert.doesNotMatch(value, /\bfill\b/);
});
