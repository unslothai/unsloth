// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// target="_blank" has nowhere to go in the Tauri webview, so every update-notes anchor goes through openLink().
// Read from source: the node suite has no DOM.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const NOTES = readSrc("components/update/release-notes-panel.tsx");

test("every _blank anchor in the release notes panel opens through openLink", () => {
  const anchors = NOTES.split("<a").slice(1).map((chunk) => chunk.slice(0, chunk.indexOf(">")));
  assert.ok(anchors.length >= 2, "expected the notes link and the summary links");
  for (const anchor of anchors) {
    if (!anchor.includes('target="_blank"')) continue;
    assert.match(anchor, /onClick=\{handleExternalClick\}/, anchor);
  }
  assert.match(NOTES, /openLink\(event\.currentTarget\.href\)/);
});
