// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One draft per model settings identity, so the two Run settings editors cannot diverge.

// Relative: the hub barrel pulls in React and the download manager.
import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "../../hub/lib/model-identity.ts";
import type { PerModelConfig } from "./per-model-config";

// Normalized like modelStorageKey, or one model spelled two ways is two drafts. The JSON pair
// also keeps "repo:quant" apart from "repo" at "quant", which a string join folds together.
function draftStorageKey(
  modelId: string,
  ggufVariant: string | null | undefined,
): string {
  return JSON.stringify([
    normalizeModelIdentity(modelId),
    normalizeGgufVariantIdentity(ggufVariant),
  ]);
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
// Editors showing each draft. It must outlive one and not the last, or a value typed and never
// applied returns as the model's settings, over a row saved elsewhere meanwhile.
const hostCounts = new Map<string, number>();
// Per draft, not per editor: opening the second host must not re-run the read and write the
// stored row back over what the first is showing.
const extraArgsHydratedByDraftKey = new Map<string, string>();

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

/** Registers one editor and returns its release; the draft and its mark live while any holds. */
export function retainModelConfigDraft(key: string): () => void {
  hostCounts.set(key, (hostCounts.get(key) ?? 0) + 1);
  let released = false;
  return () => {
    if (released) {
      return;
    }
    released = true;
    const remaining = (hostCounts.get(key) ?? 1) - 1;
    if (remaining > 0) {
      hostCounts.set(key, remaining);
      return;
    }
    hostCounts.delete(key);
    extraArgsHydratedByDraftKey.delete(key);
    if (drafts.delete(key)) {
      notify();
    }
  };
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
    savedRemember: savedRemember ?? existing.savedRemember,
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
