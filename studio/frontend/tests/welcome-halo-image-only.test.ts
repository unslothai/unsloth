// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The welcome greeting's halo darkens a background image behind the text. Glass with no image draws a
// plain backdrop in the page colour, where the halo is the same colour as what it covers: it changed
// nothing, and its blending rounded a step dark in places, drawing rings around the greeting.

import assert from "node:assert/strict";
import test from "node:test";

import postcss from "postcss";

import { WALLPAPER_HOST_FLAGS } from "../src/features/chat-background/host-flags.ts";
import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");

test("the greeting halo is drawn only where the wallpaper is an image", () => {
  const halos: postcss.Rule[] = [];
  postcss.parse(CSS).walkRules((rule) => {
    if (rule.selector.includes(".unsloth-welcome-greeting::before")) halos.push(rule);
  });
  assert.equal(halos.length, 1, "one halo rule");
  const [halo] = halos;
  for (const selector of halo.selectors) {
    assert.match(selector, /\[data-wallpaper-image\]/, `image hosts only: ${selector}`);
    assert.doesNotMatch(selector, /:has\(/, "a host flag, not a :has() re-checked on every mutation");
  }
  assert.match(halo.toString(), /radial-gradient\(ellipse, color-mix\(in srgb, var\(--background\)/);
});

test("the image flag excludes the plain Glass backdrop", () => {
  const flag = WALLPAPER_HOST_FLAGS.find(([name]) => name === "data-wallpaper-image");
  assert.ok(flag, "the host flag exists");
  assert.equal(flag[1], ":scope > .chat-wallpaper[data-ready]:not([data-glass-backdrop])");
  assert.match(readSrc("features/chat-background/backdrop.tsx"), /data-glass-backdrop=""/);
  assert.match(CSS, /\.glass-backdrop \{ background: var\(--background\); \}/, "the plain backdrop is the page colour");
});
