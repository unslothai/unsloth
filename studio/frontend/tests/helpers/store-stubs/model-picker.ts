// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export function applyPerModelConfigToRuntime(): void {}

export function currentRuntimePerModelConfig(): Record<string, unknown> {
  return {};
}

export function perModelConfigsEqual(): boolean {
  return true;
}

let residentConfig: unknown = null;

export function setResidentInitialConfig(value: unknown): void {
  residentConfig = value;
}

export function resolveResidentInitialConfig(): unknown {
  return residentConfig;
}

export function loadedContextFields(): Record<string, unknown> {
  return {};
}

export function savedContextPin(): null {
  return null;
}

export function adoptCachedRepoConfig(): null {
  return null;
}
