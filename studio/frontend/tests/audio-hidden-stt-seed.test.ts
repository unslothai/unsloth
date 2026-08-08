// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  allowedHiddenModelIdMatches,
  taskPickerRowMatches,
} from "../src/features/model-picker/components/model-selector/audio-picker-policy.ts";

const pickerSource = readFileSync(
  new URL(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
    import.meta.url,
  ),
  "utf8",
);

const row = {
  isHidden: true,
  format: "all" as const,
  matchesFormat: false,
  matchesTask: false,
  isRecommendable: false,
};

test("a hidden chat-sidecar repo remains visible when explicitly curated by its task page", () => {
  assert.equal(taskPickerRowMatches({ ...row, isCatalogSeed: true }), true);
});

test("the same hidden repo remains suppressed outside a curated task contract", () => {
  assert.equal(taskPickerRowMatches({ ...row, isCatalogSeed: false }), false);
});

test("curated hidden task ids survive the query-mode recommended filter", () => {
  assert.match(
    pickerSource,
    /const recommendedIds = useMemo[\s\S]*allowedHiddenModelIdMatches\(taskCatalogSeedIds, id\)/,
  );
});

test("an allowed hidden Audio id survives local source matching without allowing siblings", () => {
  const allowed = new Set(["unsloth/whisper-tiny"]);
  assert.equal(
    allowedHiddenModelIdMatches(
      allowed,
      "unsloth/whisper-tiny",
      "E:/models/whisper-tiny",
    ),
    true,
  );
  assert.equal(
    allowedHiddenModelIdMatches(
      allowed,
      "unsloth/whisper-base",
      "E:/models/whisper-tiny",
    ),
    false,
  );
});
