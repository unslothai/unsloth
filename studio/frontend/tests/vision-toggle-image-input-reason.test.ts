// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getImageInputUnavailableReason } from "../src/features/chat/utils/image-input-support.ts";

const VISION_GGUF = {
  id: "local/qwen3.5-4b",
  name: "Qwen3.5 4B",
  isLora: false,
  isVision: true,
  isGguf: true,
  isAudio: false,
  audioType: null,
  hasAudioInput: false,
};

function reason(overrides: Record<string, unknown> = {}) {
  return getImageInputUnavailableReason({
    activeModel: VISION_GGUF,
    isExternalModel: false,
    loadedIsMultimodal: false,
    modelLoaded: true,
    ...overrides,
  });
}

test("switching Vision off points at the switch, not at a missing mmproj", () => {
  const message = reason({ visionDisabledByUser: true });
  assert.ok(message, "attaching images should still be blocked");
  assert.match(message, /Advanced Settings/);
  assert.match(message, /Qwen3\.5 4B/);
  // The bug this branch exists to prevent: the generic copy sends someone who
  // turned the toggle off to go and find a vision model with a valid mmproj,
  // which is a hunt for a problem they do not have.
  assert.doesNotMatch(message, /valid mmproj/);
  assert.doesNotMatch(message, /Load a vision-capable model/);
});

test("a genuinely missing projector keeps the original guidance", () => {
  const message = reason({ visionDisabledByUser: false });
  assert.ok(message);
  assert.match(message, /valid mmproj/);
  assert.doesNotMatch(message, /Advanced Settings/);
});

test("an absent flag behaves exactly as before the toggle existed", () => {
  assert.equal(reason(), reason({ visionDisabledByUser: false }));
  assert.equal(reason(), reason({ visionDisabledByUser: null }));
});

test("the toggle never invents a block while the projector is loaded", () => {
  // loadedIsMultimodal wins: a stale disable_vision echo must not disable the
  // attach button for a session that did load its projector.
  assert.equal(
    reason({ loadedIsMultimodal: true, visionDisabledByUser: true }),
    null,
  );
});

test("no model loaded still reports the load, not the toggle", () => {
  const message = reason({ modelLoaded: false, visionDisabledByUser: true });
  assert.match(message ?? "", /Load a model before adding images/);
});
