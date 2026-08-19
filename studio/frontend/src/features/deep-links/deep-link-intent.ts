// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function createDeepLinkIntentGate(
  deduplicationWindowMs: number,
  now: () => number = Date.now,
) {
  let lastIntent: { key: string; handledAt: number } | null = null;
  let sequence = 0;

  return (model: string, file?: string): number | null => {
    const handledAt = now();
    const key = `${model}\0${file ?? ""}`;
    if (
      lastIntent?.key === key &&
      handledAt - lastIntent.handledAt < deduplicationWindowMs
    ) {
      return null;
    }
    lastIntent = { key, handledAt };
    sequence += 1;
    return sequence;
  };
}
