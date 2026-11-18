// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_VOICE_ORB_OVERLAY_PREFS,
  readVoiceOrbOverlayPrefs,
} from "../src/features/settings/stores/voice-orb-overlay-prefs.ts";

test("the decision arcs and caption are hidden by default", () => {
  assert.deepEqual(DEFAULT_VOICE_ORB_OVERLAY_PREFS, {
    showVoiceArcs: false,
    showVoiceCaption: false,
  });
  for (const saved of [undefined, null, {}, { showVoiceArcs: "no", showVoiceCaption: 0 }]) {
    assert.deepEqual(readVoiceOrbOverlayPrefs(saved), DEFAULT_VOICE_ORB_OVERLAY_PREFS);
  }
});

test("each toggle persists on its own", () => {
  assert.deepEqual(readVoiceOrbOverlayPrefs({ showVoiceArcs: true }), {
    showVoiceArcs: true,
    showVoiceCaption: false,
  });
  assert.deepEqual(readVoiceOrbOverlayPrefs({ showVoiceArcs: true, showVoiceCaption: false }), {
    showVoiceArcs: true,
    showVoiceCaption: false,
  });
});
