// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One in-memory draft per model settings identity so the sidebar and model-dropdown
// Run settings pages cannot diverge.

// Relative, not through the hub barrel: this module is imported by the editor and by
// node --test, and the barrel pulls in React and the download manager.
import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "../../hub/lib/model-identity.ts";
import type { PerModelConfig } from "./per-model-config";

// The same identity the settings themselves are stored under (modelStorageKey in
// ./model-identity). A Windows path spelled with either separator, a drive letter in either
// case, a trailing separator and a repo id in another case all name ONE model, so keying the
// draft on the raw text would hand the two hosts separate drafts for the model they are both
// showing. The JSON pair also keeps "repo:quant with no variant" apart from "repo with variant
// quant", which a `${id}:${variant}` join folds together.
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
// How many editors are currently showing each draft. A draft outlives one host, which is the
// point -- the sidebar copy stays mounted while collapsed and the dropdown opens over it -- but
// it must not outlive the LAST one, or a value typed and never applied would come back as the
// model's settings the next time the panel opens, and a row saved elsewhere in the meantime
// would never be read. An unmounted useState used to do that job.
const hostCounts = new Map<string, number>();
// Which server-override read has already been folded into each draft. Shared with the draft
// rather than held per editor, so opening the second host does not re-run the read and write
// the stored row back over what the first host is showing.
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

/**
 * Registers one mounted editor against a draft and returns its release. The draft, and the
 * hydration mark that goes with it, live exactly as long as some editor is showing them.
 */
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

// The three writers below need a primed draft and do nothing without one. Every editor primes
// in a layout effect before it can paint a control, and holds a retain for as long as it is
// mounted, so a write with no draft means the retain and the prime have come apart.
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
