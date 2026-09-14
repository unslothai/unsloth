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

export type ModelConfigExtraArgsEdit = {
  /** Exactly what is in the textarea, half-typed quotes included. */
  text: string;
  /** `formatExtraArgs` of the tokens this edit published, to spot an external replacement. */
  source: string;
  /**
   * The row's verdict on `text`, undefined until it has one. Shared because only the row reads
   * the raw text: an editor whose Advanced section is collapsed has no row and judges the
   * TOKENS, which `formatExtraArgs` quotes back into a balanced string, so it cannot see the
   * unfinished quote and left its Run button live over an edit the other one refuses.
   */
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
// Editors showing each draft. It must outlive one and not the last, or a value typed and never
// applied returns as the model's settings, over a row saved elsewhere meanwhile.
const hostCounts = new Map<string, number>();
// Which drafts have had the stored override row folded in, so opening the second host does not
// re-run the read and write that row back over what the first is showing. Keyed by the DRAFT and
// nothing else: the two hosts reach one model through differently SHAPED candidate lists, the
// picker carrying the load-path candidates and the sidebar only the checkpoint, so normalizing
// their spellings still leaves the lists unequal. Nothing reachable within one draft key changes
// what the read returns, and the mark dies with the draft.
const extraArgsHydratedDrafts = new Set<string>();
// What is TYPED into the Extra Arguments box, which is not what is stored: the config holds
// argv tokens. Shared for the same reason the config is: the box publishes tokens on every
// keystroke, valid or not, so a second editor re-quoted a half-typed line into balanced text,
// judged it loadable and left its Run button live over an edit the first one was refusing.
// `source` is what the config read when the edit was written, so a Reset or a hydration that
// replaces llamaExtraArgs supersedes the edit in both editors instead of being re-quoted over.
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

export function isExtraArgsHydratedForDraft(key: string): boolean {
  return extraArgsHydratedDrafts.has(key);
}

export function markExtraArgsHydratedForDraft(key: string): void {
  extraArgsHydratedDrafts.add(key);
}

export function readExtraArgsEditForDraft(
  key: string,
): ModelConfigExtraArgsEdit | undefined {
  return extraArgsEditByDraftKey.get(key);
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

/** Records the row's verdict on the edit already stored; a keystroke retires it by replacing it. */
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
