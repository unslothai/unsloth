// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveInAppShortcut } from "../src/features/voice-typing/in-app-shortcut.ts";

test("toggle mode starts on the first press and stops on the next", () => {
  const start = resolveInAppShortcut({
    action: "press",
    active: false,
    mode: "toggle",
    heldMs: 0,
    pressStarted: false,
  });
  assert.deepEqual(start, { toggle: true, pressStarted: true });
  const release = resolveInAppShortcut({
    action: "release",
    active: true,
    mode: "toggle",
    heldMs: 900,
    pressStarted: start.pressStarted,
  });
  assert.deepEqual(release, { toggle: false, pressStarted: false });
  const stop = resolveInAppShortcut({
    action: "press",
    active: true,
    mode: "toggle",
    heldMs: 0,
    pressStarted: false,
  });
  assert.deepEqual(stop, { toggle: true, pressStarted: false });
});

test("hold mode starts on press and stops on release", () => {
  assert.deepEqual(
    resolveInAppShortcut({
      action: "press",
      active: false,
      mode: "hold",
      heldMs: 0,
      pressStarted: false,
    }),
    { toggle: true, pressStarted: true },
  );
  assert.deepEqual(
    resolveInAppShortcut({
      action: "release",
      active: true,
      mode: "hold",
      heldMs: 80,
      pressStarted: true,
    }),
    { toggle: true, pressStarted: false },
  );
});

test("hybrid taps keep dictation running and holds stop it", () => {
  const tapRelease = resolveInAppShortcut({
    action: "release",
    active: true,
    mode: "hybrid",
    heldMs: 120,
    pressStarted: true,
  });
  assert.deepEqual(tapRelease, { toggle: false, pressStarted: false });
  const holdRelease = resolveInAppShortcut({
    action: "release",
    active: true,
    mode: "hybrid",
    heldMs: 650,
    pressStarted: true,
  });
  assert.deepEqual(holdRelease, { toggle: true, pressStarted: false });
});

test("a press during dictation stops it in every mode", () => {
  for (const mode of ["hold", "toggle", "hybrid"] as const) {
    const decision = resolveInAppShortcut({
      action: "press",
      active: true,
      mode,
      heldMs: 0,
      pressStarted: false,
    });
    assert.deepEqual(decision, { toggle: true, pressStarted: false });
  }
});

test("release never starts a dictation that failed to open", () => {
  assert.deepEqual(
    resolveInAppShortcut({
      action: "release",
      active: false,
      mode: "hold",
      heldMs: 800,
      pressStarted: true,
    }),
    { toggle: false, pressStarted: false },
  );
});
