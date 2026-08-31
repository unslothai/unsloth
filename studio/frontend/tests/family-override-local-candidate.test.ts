// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { isFamilyOverrideLocalCandidate } from "../src/features/model-picker/components/model-selector/family-override-local-candidate.ts";

test("an explicit family surfaces only unclassified local safetensors", () => {
  const opaqueSafetensors = { model_format: "safetensors", task: null };
  assert.equal(isFamilyOverrideLocalCandidate(opaqueSafetensors, true), true);
  assert.equal(isFamilyOverrideLocalCandidate(opaqueSafetensors, false), false);
  assert.equal(
    isFamilyOverrideLocalCandidate(
      { model_format: "safetensors", task: "text-generation" },
      true,
    ),
    false,
  );
  assert.equal(
    isFamilyOverrideLocalCandidate({ model_format: "gguf", task: null }, true),
    false,
  );
  assert.equal(
    isFamilyOverrideLocalCandidate(
      { model_format: "unknown", task: null },
      true,
    ),
    false,
  );
});

test("image and video family selectors opt all local inventories into the narrow fallback", () => {
  for (const page of [
    "../src/features/images/images-page.tsx",
    "../src/features/video/video-page.tsx",
  ]) {
    const source = readFileSync(new URL(page, import.meta.url), "utf8");
    assert.match(
      source,
      /allowUnknownLocalModels=\{familyOverride !== "auto"\}/,
    );
  }

  const picker = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal(
    picker.split("isFamilyOverrideLocalCandidate(m, allowUnknownLocalModels)")
      .length - 1,
    3,
    "LM Studio, ./models, and custom-folder rows must use the same fallback",
  );
});
