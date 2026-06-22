// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-model pre-load inference settings, persisted in localStorage so the load
// dialog can offer "Remember settings for <model>".

const KEY = "unsloth_load_settings";

export interface RememberedLoadSettings {
  contextLength: number | null;
  kvCacheDtype: string | null;
  speculativeType: string | null;
  specDraftNMax: number | null;
  tensorParallel: boolean;
}

function readAll(): Record<string, RememberedLoadSettings> {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "{}");
  } catch {
    return {};
  }
}

function writeAll(all: Record<string, RememberedLoadSettings>) {
  try {
    localStorage.setItem(KEY, JSON.stringify(all));
  } catch {
    // Ignore quota / unavailable storage.
  }
}

export function loadRememberedLoadSettings(
  modelId: string,
): RememberedLoadSettings | null {
  return readAll()[modelId] ?? null;
}

export function saveRememberedLoadSettings(
  modelId: string,
  settings: RememberedLoadSettings,
) {
  const all = readAll();
  all[modelId] = settings;
  writeAll(all);
}

export function clearRememberedLoadSettings(modelId: string) {
  const all = readAll();
  if (modelId in all) {
    delete all[modelId];
    writeAll(all);
  }
}
