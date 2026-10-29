// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Marked by the entry at startup, so code loaded later (the chat page) still dates this session from
// app start, not from when its own chunk arrived.
let startedAt: number | null = null;

export function markAppSessionStart(): void {
  startedAt ??= Date.now();
}

export function appSessionStartedAt(): number {
  startedAt ??= Date.now();
  return startedAt;
}
