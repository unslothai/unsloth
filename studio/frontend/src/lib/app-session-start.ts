// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Marked at startup so lazily loaded chunks date the session from app start.
let startedAt: number | null = null;

export function markAppSessionStart(): void {
  startedAt ??= Date.now();
}

export function appSessionStartedAt(): number {
  startedAt ??= Date.now();
  return startedAt;
}
