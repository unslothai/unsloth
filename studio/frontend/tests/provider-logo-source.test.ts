// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);
const { isDarkMonochromeIcon, resolveApiProviderLogoSrc } = await import(
  "../src/features/chat/provider-logo-source.ts"
);

const SITE_ICON = "data:image/png;base64,AAAA";
const BASE = "/studio/";

test("custom provider with a valid site icon uses the data URL", () => {
  assert.equal(resolveApiProviderLogoSrc("custom", SITE_ICON, BASE), SITE_ICON);
  assert.equal(
    resolveApiProviderLogoSrc("vllm", SITE_ICON, BASE),
    SITE_ICON,
  );
});

test("malformed or non-image site icons are ignored", () => {
  for (const bad of [
    "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
    "data:text/html;base64,PGh0bWw+",
    "https://example.com/icon.png",
    "not-a-url",
  ]) {
    assert.equal(resolveApiProviderLogoSrc("custom", bad, BASE), undefined, bad);
  }
});

test("non-custom providers never use site icons", () => {
  const src = resolveApiProviderLogoSrc("openai", SITE_ICON, BASE);
  assert.equal(src, `${BASE}hub/profile/logo/openai.svg`);
});

test("presets and hub logos keep their static assets", () => {
  assert.equal(
    resolveApiProviderLogoSrc("ollama", undefined, BASE),
    `${BASE}provider-logos/ollama.svg`,
  );
  assert.equal(
    resolveApiProviderLogoSrc("gemini", undefined, BASE),
    `${BASE}provider-logos/gemini.svg`,
  );
});

test("unknown providers stay on the generic fallback", () => {
  assert.equal(
    resolveApiProviderLogoSrc("unknown", undefined, BASE),
    undefined,
  );
});

// 32x32 RGBA icon; `paint(x, y)` returns [r, g, b, a] for each pixel.
function icon(paint: (x: number, y: number) => number[]): Uint8ClampedArray {
  const data = new Uint8ClampedArray(32 * 32 * 4);
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 32; x++) data.set(paint(x, y), (y * 32 + x) * 4);
  return data;
}
const inGlyph = (x: number, y: number) => x >= 8 && x < 24 && y >= 8 && y < 24;
const CLEAR = [0, 0, 0, 0];

test("black glyph on transparency is inverted", () => {
  assert.equal(isDarkMonochromeIcon(icon((x, y) => (inGlyph(x, y) ? [0, 0, 0, 255] : CLEAR))), true);
});

test("anti-aliased black glyph with grey edges is inverted", () => {
  const edge = (x: number, y: number) => x === 8 || x === 23 || y === 8 || y === 23;
  assert.equal(
    isDarkMonochromeIcon(
      icon((x, y) =>
        !inGlyph(x, y) ? CLEAR : edge(x, y) ? [140, 140, 140, 120] : [20, 20, 20, 255],
      ),
    ),
    true,
  );
});

test("coloured, white and empty icons are left alone", () => {
  assert.equal(isDarkMonochromeIcon(icon((x, y) => (inGlyph(x, y) ? [30, 90, 200, 255] : CLEAR))), false);
  assert.equal(isDarkMonochromeIcon(icon((x, y) => (inGlyph(x, y) ? [255, 255, 255, 255] : CLEAR))), false);
  assert.equal(isDarkMonochromeIcon(icon(() => CLEAR)), false);
  assert.equal(isDarkMonochromeIcon(icon((x, y) => (x < 2 && y < 2 ? [0, 0, 0, 255] : [0, 0, 0, 10]))), false);
});
