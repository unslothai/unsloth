// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The chain from a load response to the composer's refusal string.
 *
 * Covers response -> isMultimodalResponse -> getImageInputUnavailableReason.
 * The last hop (zustand selector into the React component) needs a browser and
 * is NOT covered here.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { isMultimodalResponse } from "../src/features/chat/types/api.ts";
import { getImageInputUnavailableReason } from "../src/features/chat/utils/image-input-support.ts";

const MODEL = {
  id: "local/qwen3.5-4b",
  name: "Qwen3.5 4B",
  isLora: false,
  isVision: true,
  isGguf: true,
  isAudio: false,
  audioType: null,
  hasAudioInput: false,
};

/** Exactly what the backend emits for a disable_vision load, verified against
 *  _llama_runtime_fields on the Python side. */
const DISABLE_VISION_RESPONSE = {
  is_vision: false,
  is_audio: false,
  audio_type: null,
  has_audio_input: false,
  vision_disabled_by_user: true,
  vision_on_cpu: false,
};

/** A Tier 2 load: the projector is attached, just running on the CPU. */
const CPU_PINNED_RESPONSE = {
  is_vision: true,
  is_audio: false,
  audio_type: null,
  has_audio_input: false,
  vision_disabled_by_user: false,
  vision_on_cpu: true,
};

function reasonFor(response: typeof DISABLE_VISION_RESPONSE) {
  return getImageInputUnavailableReason({
    activeModel: MODEL,
    isExternalModel: false,
    loadedIsMultimodal: isMultimodalResponse(response),
    modelLoaded: true,
    visionDisabledByUser: response.vision_disabled_by_user,
  });
}

test("a disable_vision response blocks images and names the switch", () => {
  assert.equal(isMultimodalResponse(DISABLE_VISION_RESPONSE), false);

  assert.equal(
    reasonFor(DISABLE_VISION_RESPONSE),
    "Vision is turned off for Qwen3.5 4B. Turn it back on in the model's Advanced Settings to attach images.",
  );
});

test("a CPU-pinned projector still accepts images", () => {
  // Tier 2 trades speed for VRAM. It must not cost the user the capability.
  assert.equal(isMultimodalResponse(CPU_PINNED_RESPONSE), true);
  assert.equal(reasonFor(CPU_PINNED_RESPONSE), null);
});

test("an ordinary vision load is unchanged by the new fields", () => {
  const legacy = {
    is_vision: true,
    is_audio: false,
    audio_type: null,
    has_audio_input: false,
    // A backend from before this change sends neither field.
    vision_disabled_by_user: undefined as unknown as boolean,
    vision_on_cpu: undefined as unknown as boolean,
  };

  assert.equal(isMultimodalResponse(legacy), true);
  assert.equal(reasonFor(legacy), null);
});

test("a missing projector still gets the find-an-mmproj guidance", () => {
  const noProjector = {
    ...DISABLE_VISION_RESPONSE,
    vision_disabled_by_user: false,
  };

  assert.match(reasonFor(noProjector) ?? "", /valid mmproj/);
});
