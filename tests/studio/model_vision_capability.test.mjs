// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isKnownTextOnlySelection,
  normalizeGgufVisionCapability,
} from "../../studio/frontend/src/features/chat/utils/model-vision-capability.ts";

test("undefined catalog target is not treated as text-only", () => {
  assert.equal(isKnownTextOnlySelection({}, undefined), false);
});

test("known vision GGUF suppresses the historical-image warning", () => {
  assert.equal(
    isKnownTextOnlySelection(
      { isGguf: true, isVision: true },
      { isGguf: true, isVision: false },
    ),
    false,
  );
});

test("known text-only GGUF emits the historical-image warning", () => {
  assert.equal(
    isKnownTextOnlySelection(
      { isGguf: true, isVision: false },
      { isGguf: true, isVision: false },
    ),
    true,
  );
});

test("unknown GGUF capability does not trust a stale catalog false", () => {
  assert.equal(
    isKnownTextOnlySelection(
      { isGguf: true },
      { isGguf: true, isVision: false },
    ),
    false,
  );
});
test("missing and malformed GGUF vision metadata remain unknown", () => {
  for (const value of [undefined, null, "false", 0]) {
    const isVision = normalizeGgufVisionCapability(value);
    assert.equal(isVision, undefined);
    assert.equal(
      isKnownTextOnlySelection({ isGguf: true, isVision }),
      false,
    );
  }
  assert.equal(normalizeGgufVisionCapability(false), false);
  assert.equal(
    isKnownTextOnlySelection({ isGguf: true, isVision: false }),
    true,
  );
});



test("known text-only non-GGUF still emits the warning", () => {
  assert.equal(
    isKnownTextOnlySelection({}, { isGguf: false, isVision: false }),
    true,
  );
});
