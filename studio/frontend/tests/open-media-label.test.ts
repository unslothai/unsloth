// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Images preview is a button that opens the viewer, and its name replaces the image's alt, so
// it still has to say what the picture is.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { shortPrompt } from "../src/lib/open-media-label.ts";

test("the prompt is kept, whitespace collapsed", () => {
  assert.equal(shortPrompt("a red fox in snow"), "a red fox in snow");
  assert.equal(shortPrompt("  a red\n fox  "), "a red fox");
});

test("no prompt gives nothing, for the plain action", () => {
  assert.equal(shortPrompt(""), "");
  assert.equal(shortPrompt(" \n "), "");
});

test("a long prompt is cut at a word, with an ellipsis", () => {
  const prompt = "a very detailed painting of a lighthouse on a cliff at dusk, waves crashing below, gulls";
  assert.equal(shortPrompt(prompt, 40), "a very detailed painting of a lighthouse…");
  assert.equal(shortPrompt(prompt, 38), "a very detailed painting of a…");
  // One long word has no space to cut at, so it is cut where the limit falls.
  assert.equal(shortPrompt("x".repeat(50), 10), `${"x".repeat(10)}…`);
});

test("the cut counts code points, so an emoji is never split", () => {
  assert.equal(shortPrompt(`a${"🐱".repeat(20)}`, 10), `a${"🐱".repeat(9)}…`);
});

test("the Images preview is named for its prompt", () => {
  const page = readSrc("features/images/images-page.tsx");
  const preview = page.slice(
    page.indexOf("                  src={selectedSrc}\n                  alt={selected.prompt}"),
    page.indexOf("/>", page.indexOf("                  alt={selected.prompt}")),
  );
  assert.ok(preview.includes('role="button"'));
  assert.ok(preview.includes("tabIndex={0}"));
  assert.ok(preview.includes("aria-label={openImageLabel(t, selected.prompt)}"));
});
