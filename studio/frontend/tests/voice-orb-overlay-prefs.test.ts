// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_VOICE_ORB_OVERLAY_PREFS,
  readVoiceOrbOverlayPrefs,
} from "../src/features/settings/stores/voice-orb-overlay-prefs.ts";

import { readSrc } from "./helpers/kit.ts";

test("the decision arcs and caption are shown by default", () => {
  assert.deepEqual(DEFAULT_VOICE_ORB_OVERLAY_PREFS, {
    showVoiceArcs: true,
    showVoiceCaption: true,
  });
  // A store saved before these settings existed, or a corrupt one, keeps both on.
  for (const saved of [undefined, null, {}, { showVoiceArcs: "no", showVoiceCaption: 0 }]) {
    assert.deepEqual(readVoiceOrbOverlayPrefs(saved), DEFAULT_VOICE_ORB_OVERLAY_PREFS);
  }
});

test("each toggle persists on its own", () => {
  assert.deepEqual(readVoiceOrbOverlayPrefs({ showVoiceArcs: false }), {
    showVoiceArcs: false,
    showVoiceCaption: true,
  });
  assert.deepEqual(readVoiceOrbOverlayPrefs({ showVoiceArcs: true, showVoiceCaption: false }), {
    showVoiceArcs: true,
    showVoiceCaption: false,
  });
});

test("the voice settings store starts from the defaults and restores saved values", () => {
  const store = readSrc("features/settings/stores/voice-settings-store.ts");
  assert.match(store, /\.\.\.DEFAULT_VOICE_ORB_OVERLAY_PREFS,/);
  assert.match(store, /\.\.\.readVoiceOrbOverlayPrefs\(saved\),/);
  const orb = readSrc("components/assistant-ui/voice-orb.tsx");
  assert.match(orb, /showOverlay && showArcs && <VoiceDecisionRing \/>/);
  assert.match(orb, /chip: showOverlay && showCaption && hasDecision/);
});
