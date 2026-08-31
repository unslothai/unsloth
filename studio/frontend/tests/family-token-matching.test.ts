// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { familyTokenMatches } from "../src/lib/family-token-matching.ts";

test("frontend family matching mirrors backend delimiter flexibility", () => {
  for (const identifier of [
    "custom/qwen-image-edit",
    "custom/qwen.image.edit",
    "custom/qwen_image_edit",
  ]) {
    assert.equal(familyTokenMatches("qwen-image-edit", identifier), true, identifier);
  }
  assert.equal(familyTokenMatches("minimax-h3", "custom/minimax.h3"), true);
  assert.equal(familyTokenMatches("minimax-h3", "custom/minimaximal.h3"), false);
});

test("image-edit and H3 workflow gates use the shared matcher", () => {
  const picker = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(picker, /return familyTokenMatches\(keyword, id\)/);

  const videoRouting = readFileSync(
    new URL(
      "../src/features/video/video-generation-defaults.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const start = videoRouting.indexOf("function isH3PipelinePick(");
  const end = videoRouting.indexOf("const MODEL_DEFAULTS", start);
  assert.ok(start >= 0 && end > start);
  assert.match(
    videoRouting.slice(start, end),
    /familyTokenMatches\("minimax-h3", leaf\)/,
  );
});
