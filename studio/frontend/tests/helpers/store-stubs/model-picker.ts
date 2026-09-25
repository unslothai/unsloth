// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The picker barrel reaches the whole chat UI; the store only needs these three.

export function applyPerModelConfigToRuntime(): void {}

export function currentRuntimePerModelConfig(): Record<string, unknown> {
  return {};
}

export function perModelConfigsEqual(): boolean {
  return true;
}
