// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Images preview is a button that opens the viewer, and its name replaces the image's alt, so
// it still has to say what the picture is.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { openMediaLabel } from "../src/lib/open-media-label.ts";

test("the label keeps the prompt", () => {
  assert.equal(openMediaLabel("image", "a red fox in snow"), "Open image: a red fox in snow");
  assert.equal(openMediaLabel("image", "  a red\n fox  "), "Open image: a red fox");
});

test("no prompt leaves the plain action", () => {
  assert.equal(openMediaLabel("image", ""), "Open image");
  assert.equal(openMediaLabel("video", " \n "), "Open video");
});

test("a long prompt is cut at a word, with an ellipsis", () => {
  const prompt = "a very detailed painting of a lighthouse on a cliff at dusk, waves crashing below, gulls";
  const label = openMediaLabel("image", prompt, 40);
  assert.equal(label, "Open image: a very detailed painting of a lighthouse…");
  assert.equal(openMediaLabel("image", prompt, 38), "Open image: a very detailed painting of a…");
  // One long word has no space to cut at, so it is cut where the limit falls.
  assert.equal(openMediaLabel("image", "x".repeat(50), 10), `Open image: ${"x".repeat(10)}…`);
});

test("the cut counts code points, so an emoji is never split", () => {
  const label = openMediaLabel("image", `a${"🐱".repeat(20)}`, 10);
  assert.equal(label, `Open image: a${"🐱".repeat(9)}…`);
});

test("the Images preview is named for its prompt", () => {
  const page = readSrc("features/images/images-page.tsx");
  const preview = page.slice(
    page.indexOf("                  src={selectedSrc}\n                  alt={selected.prompt}"),
    page.indexOf("/>", page.indexOf("                  alt={selected.prompt}")),
  );
  assert.ok(preview.includes('role="button"'));
  assert.ok(preview.includes("tabIndex={0}"));
  assert.ok(preview.includes('aria-label={openMediaLabel("image", selected.prompt)}'));
});
