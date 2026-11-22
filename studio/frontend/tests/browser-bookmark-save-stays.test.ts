// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("saving a bookmark from the star does not submit the address bar", () => {
  const panel = readSrc("features/browser/browser-panel.tsx");
  assert.match(panel, /<AddressBar[\s\S]*?actions=\{[\s\S]*?<BookmarkStar /);
  const editor = readSrc("features/browser/bookmarks.tsx");
  const submit = editor.match(/function BookmarkEditor\([\s\S]*?<form[\s\S]*?onSubmit=\{\(event\) => \{([\s\S]*?)\n\s*\}\}/);
  assert.ok(submit, "the bookmark editor has a submit handler");
  assert.match(submit[1], /event\.stopPropagation\(\);/);
});
