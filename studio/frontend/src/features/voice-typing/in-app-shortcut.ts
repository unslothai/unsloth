// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { VoiceTypingMode } from "./voice-typing-store";

/** Matches the native poller's TAP_THRESHOLD: shorter is a tap in hybrid mode. */
export const IN_APP_TAP_THRESHOLD_MS = 300;

export interface InAppShortcutInput {
  action: "press" | "release";
  /** Whether an in-app composer dictation is running right now. */
  active: boolean;
  mode: VoiceTypingMode;
  heldMs: number;
  /** Whether this press was the one that started the dictation. */
  pressStarted: boolean;
}

export interface InAppShortcutDecision {
  /** Run the composer's start/stop handler (it toggles on its live state). */
  toggle: boolean;
  /** Value to keep for the matching release. */
  pressStarted: boolean;
}

/**
 * Maps a global chord edge to the composer's own dictation toggle, preserving
 * the three activation modes:
 * - toggle: every press starts or stops;
 * - hybrid: a tap starts and stays on, a hold stops on release;
 * - hold: press starts, release stops.
 * Pressing while dictation is already running always stops it.
 */
export function resolveInAppShortcut(
  input: InAppShortcutInput,
): InAppShortcutDecision {
  if (input.action === "press") {
    return { toggle: true, pressStarted: !input.active };
  }
  if (!input.pressStarted) return { toggle: false, pressStarted: false };
  if (input.mode === "toggle") return { toggle: false, pressStarted: false };
  if (input.mode === "hybrid" && input.heldMs < IN_APP_TAP_THRESHOLD_MS) {
    // A tap leaves dictation running, exactly like the in-app shortcut.
    return { toggle: false, pressStarted: false };
  }
  // Only stop a session this press actually started; if starting failed,
  // toggling on release would start a phantom dictation.
  return { toggle: input.active, pressStarted: false };
}
