// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One in-memory draft per model settings identity so the sidebar and model-dropdown
// Run settings pages cannot diverge.

import type { PerModelConfig } from "./per-model-config";

function draftStorageKey(
  modelId: string,
  ggufVariant: string | null | undefined,
): string {
  return ggufVariant ? `${modelId}:${ggufVariant}` : modelId;
}

export type ModelConfigDraftSnapshot = {
  config: PerModelConfig;
  remember: boolean;
  savedRemember: boolean;
  /** Last `loadedConfigSignature` applied to this draft from the resident process. */
  appliedLiveSignature: string;
};

const drafts = new Map<string, ModelConfigDraftSnapshot>();
const listeners = new Set<() => void>();

/** Stable React key: model + quant only. Live config sync goes through the draft store. */
export function modelConfigEditorKey(
  modelId: string,
  ggufVariant: string | null | undefined,
): string {
  return draftStorageKey(modelId, ggufVariant);
}

export function modelConfigDraftKey(
  configId: string,
  ggufVariant: string | null | undefined,
): string {
  return draftStorageKey(configId, ggufVariant);
}

function notify() {
  for (const listener of listeners) {
    listener();
  }
}

export function subscribeModelConfigDraft(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function readModelConfigDraft(
  key: string,
): ModelConfigDraftSnapshot | undefined {
  return drafts.get(key);
}

export function primeModelConfigDraft(
  key: string,
  seed: { config: PerModelConfig; remembered: boolean },
  liveSignature: string,
): ModelConfigDraftSnapshot {
  const existing = drafts.get(key);
  if (!existing) {
    const created: ModelConfigDraftSnapshot = {
      config: seed.config,
      remember: seed.remembered,
      savedRemember: seed.remembered,
      appliedLiveSignature: liveSignature,
    };
    drafts.set(key, created);
    notify();
    return created;
  }
  const liveArrived =
    liveSignature !== "none" &&
    existing.appliedLiveSignature !== liveSignature;
  if (liveArrived) {
    const next: ModelConfigDraftSnapshot = {
      config: seed.config,
      remember: seed.remembered,
      savedRemember: seed.remembered,
      appliedLiveSignature: liveSignature,
    };
    drafts.set(key, next);
    notify();
    return next;
  }
  return existing;
}

export function replaceModelConfigDraft(
  key: string,
  config: PerModelConfig,
  options?: {
    remember?: boolean;
    savedRemember?: boolean;
    appliedLiveSignature?: string;
  },
): void {
  const existing = drafts.get(key);
  const next: ModelConfigDraftSnapshot = {
    config,
    remember: options?.remember ?? existing?.remember ?? false,
    savedRemember: options?.savedRemember ?? existing?.savedRemember ?? false,
    appliedLiveSignature:
      options?.appliedLiveSignature ??
      existing?.appliedLiveSignature ??
      "none",
  };
  drafts.set(key, next);
  notify();
}

export function patchModelConfigDraft(
  key: string,
  patch:
    | Partial<PerModelConfig>
    | ((current: PerModelConfig) => PerModelConfig),
): void {
  const existing = drafts.get(key);
  if (!existing) {
    return;
  }
  const nextConfig =
    typeof patch === "function"
      ? patch(existing.config)
      : { ...existing.config, ...patch };
  drafts.set(key, { ...existing, config: nextConfig });
  notify();
}

export function setModelConfigDraftRemember(
  key: string,
  remember: boolean,
  savedRemember?: boolean,
): void {
  const existing = drafts.get(key);
  if (!existing) {
    return;
  }
  drafts.set(key, {
    ...existing,
    remember,
    savedRemember: savedRemember ?? remember,
  });
  notify();
}

export function setModelConfigDraftSavedRemember(
  key: string,
  savedRemember: boolean,
): void {
  const existing = drafts.get(key);
  if (!existing) {
    return;
  }
  drafts.set(key, { ...existing, savedRemember });
  notify();
}

const extraArgsHydratedByDraftKey = new Map<string, string>();

export function extraArgsHydrationIdentityForDraft(
  key: string,
): string | null {
  return extraArgsHydratedByDraftKey.get(key) ?? null;
}

export function markExtraArgsHydratedForDraft(
  key: string,
  identity: string,
): void {
  extraArgsHydratedByDraftKey.set(key, identity);
}

export function resetExtraArgsHydrationForDraft(key: string): void {
  extraArgsHydratedByDraftKey.delete(key);
}
