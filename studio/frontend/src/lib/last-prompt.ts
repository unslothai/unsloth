// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Storage failures read as nothing saved. */
// "unsloth" keys, so another account signing in clears them (transitionBrowserAccount).
const PREFIX = "unsloth_last_prompt:";
const DISMISSED_PREFIX = "unsloth_example_prompt_dismissed:";

export function readLastPrompt(key: string): string {
  try {
    return localStorage.getItem(PREFIX + key) ?? "";
  } catch {
    return "";
  }
}

export function saveLastPrompt(key: string, prompt: string): void {
  try {
    localStorage.setItem(PREFIX + key, prompt);
  } catch {
    // storage unavailable
  }
}

export function isExampleDismissed(key: string): boolean {
  try {
    return localStorage.getItem(DISMISSED_PREFIX + key) !== null;
  } catch {
    return false;
  }
}

export function dismissExample(key: string): void {
  try {
    localStorage.setItem(DISMISSED_PREFIX + key, "1");
  } catch {
    // storage unavailable
  }
}
