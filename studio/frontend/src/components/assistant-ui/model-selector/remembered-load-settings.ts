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

// Storage key for a pick's remembered settings. The remembered knobs are
// VRAM-budget driven (context override, KV-cache dtype, tensor-parallel), so the
// right values differ per quant. An HF repo collapses all its GGUF variants into
// one `id`, so fold the variant in to scope settings per quant. Local .gguf
// paths key by their file path (already file-specific); native drag-drop files
// key by display label, so same-named files in different folders share an entry.
export function rememberedLoadSettingsKey(selection: {
  id: string;
  ggufVariant?: string | null;
}): string {
  return selection.ggufVariant
    ? `${selection.id}::${selection.ggufVariant}`
    : selection.id;
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
  key: string,
): RememberedLoadSettings | null {
  return readAll()[key] ?? null;
}

export function saveRememberedLoadSettings(
  key: string,
  settings: RememberedLoadSettings,
) {
  const all = readAll();
  all[key] = settings;
  writeAll(all);
}

export function clearRememberedLoadSettings(key: string) {
  const all = readAll();
  if (key in all) {
    delete all[key];
    writeAll(all);
  }
}
