// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-model pre-load inference settings, persisted in localStorage so the load
// dialog can offer "Remember settings for <model>". GGUF picks only: every
// field is a llama.cpp load knob, so all save/restore call sites gate on
// GGUF-ness (a non-GGUF blob would only snapshot leftover standing values).

const KEY = "unsloth_load_settings";

export interface RememberedLoadSettings {
  contextLength: number | null;
  kvCacheDtype: string | null;
  speculativeType: string | null;
  specDraftNMax: number | null;
  tensorParallel: boolean;
  // GPU Memory controls. Optional so an older blob (which lacked them) still
  // parses, leaving the live knobs untouched on apply. The mode is kept with the
  // manual knobs (gpuLayers/nCpuMoe are ignored outside Manual mode). A null
  // selectedGpuIds is meaningful (all GPUs), so it's distinguished from absent.
  // The per-GPU split ratio is deliberately NOT remembered: it's positionally
  // bound to the exact GPU set/order and unvalidated, so it would mismatch.
  gpuMemoryMode?: "auto" | "manual";
  gpuLayers?: number;
  nCpuMoe?: number;
  selectedGpuIds?: number[] | null;
}

// Storage key for a pick's remembered settings, scoped per quant (the VRAM-budget
// knobs differ per quant). An HF repo collapses its GGUF variants into one `id`,
// so fold the variant in. Local .gguf paths are already file-specific; native
// drag-drop files key by display label, so same-named files share an entry.
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
