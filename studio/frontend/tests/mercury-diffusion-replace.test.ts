// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mercury diffusion streaming rewrites the reply in place: each `_diffusionReplace` is the whole
// text so far, so the adapter swaps it in rather than appending, and the closing delta.content
// (the same full text) lands once.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { createSegmentedAssistantText } = await import(
  "../src/features/chat/utils/incremental-assistant-content.ts"
);

const src = readSrc("features/chat/api/chat-adapter.ts");

test("the replace snapshot is read as the chunk's text", () => {
  assert.match(
    src,
    /typeof diffusionReplace === "string"\s*\? diffusionReplace\s*: chunk\.choices\?\.\[0\]\?\.delta\?\.content/,
  );
});

test("a diffusing run swaps the snapshot in, the closing delta.content included", () => {
  assert.match(
    src,
    /typeof diffusionReplace === "string" \|\|\s*diffusion\.active\s*\) \{\s*rewroteText = replaceDiffusionSnapshot\(delta\);\s*\} else \{\s*appendCumulative\(delta\);/,
  );
  const at = src.indexOf("const replaceDiffusionSnapshot = ");
  assert.ok(at > 0, "the replace helper disappeared");
  const helper = src.slice(at, src.indexOf("\n      };", at));
  // The identical closing text is a no-op, so the saved reply holds the final text exactly once.
  assert.match(helper, /const cut = diffusion\.cutAt\(cumulativeText\.length, text\);\s*if \(cut === null\) return false;/);
  assert.match(helper, /cumulativeText = cumulativeText\.slice\(0, cut\);/);
  assert.match(helper, /thinkTags\.retract\(cumulativeText\);\s*placeholderWatch\.retract\(cumulativeText\);/);
  assert.match(helper, /segmentedText = createSegmentedAssistantText\(/);
  assert.match(helper, /appendCumulative\(text\);\s*return true;/);
});

test("a same-length rewrite still publishes", () => {
  assert.match(
    src,
    /cumulativeText\.length === textLenBeforeChunk &&\s*!rewroteText/,
  );
});

test("the replace state is per run, so regenerate starts appending again", () => {
  assert.match(src, /const diffusion = createDiffusionReplace\(\);/);
});

test("a fresh parse after the swap shows the reasoning and only the newest snapshot", () => {
  const reasoning = "<think>plan</think>";
  const first = createSegmentedAssistantText();
  first.appendText(reasoning + "Hxllo");
  first.runs(reasoning + "Hxllo", []);
  // What the replace path does: new parser, only the new snapshot appended, full buffer passed.
  const swapped = createSegmentedAssistantText();
  swapped.appendText("Hello");
  const parts = JSON.stringify(swapped.runs(reasoning + "Hello", []));
  assert.match(parts, /plan/);
  assert.match(parts, /Hello/);
  assert.doesNotMatch(parts, /Hxllo/);
});
