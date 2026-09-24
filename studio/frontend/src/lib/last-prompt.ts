// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The last prompt a media page generated with, kept across reloads. Storage failures fall back to
 * the page's example, so the box is never left without one.
 */
const PREFIX = "unsloth_last_prompt:";

export function readLastPrompt(key: string, fallback: string): string {
  try {
    return localStorage.getItem(PREFIX + key) ?? fallback;
  } catch {
    return fallback;
  }
}

export function saveLastPrompt(key: string, prompt: string): void {
  try {
    localStorage.setItem(PREFIX + key, prompt);
  } catch {
    // storage unavailable
  }
}
