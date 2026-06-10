// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const LOCAL_MODEL_SOURCE = {
  MODELS_DIR: "models_dir",
  HF_CACHE: "hf_cache",
  LMSTUDIO: "lmstudio",
  OLLAMA: "ollama",
  CUSTOM: "custom",
} as const;

export const LOCAL_MODEL_SOURCES = [
  LOCAL_MODEL_SOURCE.MODELS_DIR,
  LOCAL_MODEL_SOURCE.HF_CACHE,
  LOCAL_MODEL_SOURCE.LMSTUDIO,
  LOCAL_MODEL_SOURCE.OLLAMA,
  LOCAL_MODEL_SOURCE.CUSTOM,
] as const;
export type LocalSource = (typeof LOCAL_MODEL_SOURCES)[number];

export const INVENTORY_HINT_KIND = {
  MODEL: "model",
  GGUF: "gguf",
  DATASET: "dataset",
} as const;

export const INVENTORY_HINT_KINDS = [
  INVENTORY_HINT_KIND.MODEL,
  INVENTORY_HINT_KIND.GGUF,
  INVENTORY_HINT_KIND.DATASET,
] as const;
export type InventoryHintKind = (typeof INVENTORY_HINT_KINDS)[number];
