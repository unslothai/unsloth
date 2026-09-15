// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One draft per model settings identity, so the two Run settings editors cannot diverge.

// Relative: the hub barrel pulls in React and the download manager.
import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "../../hub/lib/model-identity.ts";
import type { PerModelConfig } from "./per-model-config";

// Normalized like modelStorageKey; the JSON pair keeps "repo:quant" apart from "repo" at "quant".
function draftStorageKey(
  modelId: string,
  ggufVariant: string | null | undefined,
): string {
  return JSON.stringify([
    normalizeModelIdentity(modelId),
    normalizeGgufVariantIdentity(ggufVariant),
  ]);
}

export type ModelConfigExtraArgsEdit = {
  /** Exactly what is in the textarea, half-typed quotes included. */
  text: string;
  /** `formatExtraArgs` of the tokens this edit published, to spot an external replacement. */
  source: string;
  /** The row's verdict on `text`; an editor with Advanced collapsed has no row to ask. */
  loadable?: boolean;
};

export type ModelConfigDraftSnapshot = {
  config: PerModelConfig;
  remember: boolean;
  savedRemember: boolean;
  /** Last `loadedConfigSignature` applied to this draft from the resident process. */
  appliedLiveSignature: string;
};

const drafts = new Map<string, ModelConfigDraftSnapshot>();
const listeners = new Set<() => void>();
// Editors showing each draft: it must outlive one and not the last.
const hostCounts = new Map<string, number>();
// Drafts whose stored override row is folded in. Keyed by the DRAFT, never by the candidate
// keys: the two hosts build differently SHAPED lists, so no normalizing makes them equal.
const extraArgsHydratedDrafts = new Set<string>();
// Drafts the USER changed, which the config alone cannot tell: a read may neither replace an
// edited draft nor re-run over one, since the peer's edit is already in its configAtStart.
const editedDrafts = new Set<string>();
// What is TYPED into the Extra Arguments box; the config holds argv tokens. Shared, or a second
// editor re-quotes a half-typed line into balanced text and judges it loadable.
const extraArgsEditByDraftKey = new Map<string, ModelConfigExtraArgsEdit>();

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
  // A fresh editor re-reads the row: the sidebar host never unmounts while a model is resident,
  // so a permanent mark hid settings another origin saved. Never over an edit.
  if (!editedDrafts.has(key)) {
    extraArgsHydratedDrafts.delete(key);
  }
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
    extraArgsHydratedDrafts.delete(key);
    editedDrafts.delete(key);
    extraArgsEditByDraftKey.delete(key);
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
    editedDrafts.delete(key);
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
    editedDrafts.delete(key);
    // Re-seeded from the resident process, as external as a hydration.
    extraArgsEditByDraftKey.delete(key);
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
  editedDrafts.delete(key);
  // Token equality cannot tell an A -> B -> A round trip from no change, so the edit goes with
  // the value it described rather than waiting to be superseded.
  extraArgsEditByDraftKey.delete(key);
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

export function isExtraArgsHydratedForDraft(key: string): boolean {
  return extraArgsHydratedDrafts.has(key);
}

export function markExtraArgsHydratedForDraft(key: string): void {
  extraArgsHydratedDrafts.add(key);
}

export function markModelConfigDraftEdited(key: string): void {
  editedDrafts.add(key);
}

export function clearModelConfigDraftEdited(key: string): void {
  editedDrafts.delete(key);
}

export function isModelConfigDraftEdited(key: string): boolean {
  return editedDrafts.has(key);
}

export function readExtraArgsEditForDraft(
  key: string,
): ModelConfigExtraArgsEdit | undefined {
  return extraArgsEditByDraftKey.get(key);
}

/** Retires the raw edit, so a replacement from outside the box is not re-quoted over. */
export function clearExtraArgsEditForDraft(key: string): boolean {
  return extraArgsEditByDraftKey.delete(key);
}

export function setExtraArgsEditForDraft(
  key: string,
  edit: ModelConfigExtraArgsEdit,
): void {
  const existing = extraArgsEditByDraftKey.get(key);
  if (
    existing &&
    existing.text === edit.text &&
    existing.source === edit.source &&
    existing.loadable === edit.loadable
  ) {
    return;
  }
  extraArgsEditByDraftKey.set(key, edit);
  notify();
}

/** Records the row's verdict; a keystroke retires it by replacing the edit. */
export function setExtraArgsEditLoadableForDraft(
  key: string,
  loadable: boolean,
): void {
  const existing = extraArgsEditByDraftKey.get(key);
  if (!existing || existing.loadable === loadable) {
    return;
  }
  extraArgsEditByDraftKey.set(key, { ...existing, loadable });
  notify();
}
