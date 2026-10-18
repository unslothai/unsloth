// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_VOICE_TYPING_SHORTCUT,
  chordFromEvent,
  formatShortcut,
  isModifierCode,
  isValidShortcut,
  keyFromCode,
} from "../src/features/voice-typing/shortcut.ts";

const event = (
  code: string,
  modifiers: Partial<Pick<KeyboardEvent, "ctrlKey" | "shiftKey" | "altKey" | "metaKey">> = {},
) => ({
  code,
  ctrlKey: false,
  shiftKey: false,
  altKey: false,
  metaKey: false,
  ...modifiers,
});

test("capture builds canonical chords from key events", () => {
  assert.equal(
    chordFromEvent(event("Space", { ctrlKey: true, shiftKey: true })),
    "Ctrl+Shift+Space",
  );
  assert.equal(chordFromEvent(event("KeyQ", { altKey: true })), "Alt+Q");
  assert.equal(chordFromEvent(event("Period", { metaKey: true })), "Super+.");
  assert.equal(chordFromEvent(event("Digit7", { ctrlKey: true })), "Ctrl+7");
  assert.equal(chordFromEvent(event("F4", { ctrlKey: true })), "Ctrl+F4");
});

test("modifier-only presses keep waiting and bare keys are refused", () => {
  assert.equal(chordFromEvent(event("ControlLeft", { ctrlKey: true })), null);
  assert.equal(chordFromEvent(event("ShiftRight", { shiftKey: true })), null);
  assert.equal(isModifierCode("ControlLeft"), true);
  assert.equal(isModifierCode("KeyA"), false);
  // No modifier: would shadow typing everywhere.
  assert.equal(chordFromEvent(event("KeyA")), null);
  assert.equal(chordFromEvent(event("Space")), null);
  // Unknown / unusable keys.
  assert.equal(chordFromEvent(event("Numpad1", { ctrlKey: true })), null);
  assert.equal(chordFromEvent(event("Escape", { ctrlKey: true })), null);
});

test("key labels map the codes the native parser understands", () => {
  assert.equal(keyFromCode("KeyA"), "A");
  assert.equal(keyFromCode("Digit7"), "7");
  assert.equal(keyFromCode("F4"), "F4");
  assert.equal(keyFromCode("Semicolon"), ";");
  assert.equal(keyFromCode("Backquote"), "`");
  assert.equal(keyFromCode("Numpad1"), null);
  assert.equal(keyFromCode("F25"), null);
});

test("shortcut validation matches the native parser's rules", () => {
  assert.equal(isValidShortcut(DEFAULT_VOICE_TYPING_SHORTCUT), true);
  assert.equal(isValidShortcut("Alt+F4"), true);
  assert.equal(isValidShortcut("Super+."), true);
  assert.equal(isValidShortcut("Ctrl+Shift"), false);
  assert.equal(isValidShortcut("Space"), false);
  assert.equal(isValidShortcut("Ctrl+Escape"), false);
  assert.equal(isValidShortcut("Ctrl+A+B"), false);
});

test("display formatting spells out the chord", () => {
  assert.equal(formatShortcut("Ctrl+Shift+Space"), "Ctrl + Shift + Space");
  assert.equal(formatShortcut("Super+A"), "Win + A");
});
