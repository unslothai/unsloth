// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * True when the prompt uses a second-precision time variable ({{$now}} or
 * {{$time}}). Those are re-filled on every request, so the prompt prefix
 * changes each turn and prefix caching never matches — every message pays a
 * full prefill (#9177: 55s vs 0.95s first token). {{$date}} flips once per
 * day and is usually acceptable.
 */
export function promptUsesHighPrecisionTimeVariables(prompt: string): boolean {
  if (!prompt) {
    return false;
  }
  return /{{\s*\$now\s*}}|{{\s*\$time\s*}}/i.test(prompt);
}
