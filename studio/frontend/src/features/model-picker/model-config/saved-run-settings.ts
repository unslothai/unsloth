// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";
import { syncModelOverride } from "../api/model-overrides";
import type { ModelPickTarget } from "../components/model-selector/types";
import {
  type ModelConfigDraftSnapshot,
  isModelConfigDraftEdited,
  modelConfigDraftKey,
  readModelConfigDraft,
  replaceModelConfigDraft,
  setModelConfigDraftRemember,
} from "./model-config-draft";
import { modelStorageKey } from "./model-identity";
import {
  PER_MODEL_CONFIG_STORAGE_KEY,
  PER_MODEL_CONFIG_UPDATED_EVENT,
  type PerModelConfig,
  deletePerModelConfig,
  resolveInitialConfig,
  savePerModelConfig,
} from "./per-model-config";

// Every picker row asks, so answers are cached until the next write rather than re-parsing the map per row.
let savedByKey = new Map<string, PerModelConfig | null>();
const listeners = new Set<() => void>();
let detach: (() => void) | null = null;
// What the cache was read from while unsubscribed, when only a re-read can notice a write.
let cachedRaw: string | null | undefined;

function invalidate(): void {
  savedByKey = new Map();
  cachedRaw = undefined;
}

function storedRaw(): string | null {
  try {
    return localStorage.getItem(PER_MODEL_CONFIG_STORAGE_KEY);
  } catch {
    return null;
  }
}

function settingsKey(target: ModelPickTarget): [string, string | null] {
  return [target.configId ?? target.id, target.ggufVariant ?? null];
}

/** The settings this row would load with, or null when nothing is saved for it. */
export function savedRunSettings(
  target: ModelPickTarget,
): PerModelConfig | null {
  const [id, variant] = settingsKey(target);
  const key = modelStorageKey(id, variant);
  // Cached either way, since useSyncExternalStore needs a stable snapshot before it subscribes.
  if (!detach) {
    const raw = storedRaw();
    if (raw !== cachedRaw) {
      invalidate();
      cachedRaw = raw;
    }
  }
  const cached = savedByKey.get(key);
  if (cached !== undefined) {
    return cached;
  }
  const resolved = resolveInitialConfig(id, variant);
  const value = resolved.remembered ? resolved.config : null;
  savedByKey.set(key, value);
  if (!detach) {
    // The first read can migrate legacy records into the store; that is not a new write.
    cachedRaw = storedRaw();
  }
  return value;
}

function notify(): void {
  invalidate();
  for (const listener of [...listeners]) {
    listener();
  }
}

export function subscribeSavedRunSettings(onChange: () => void): () => void {
  listeners.add(onChange);
  if (!detach && typeof window?.addEventListener === "function") {
    // Writes made while nobody listened never invalidated the cache; an unchanged store keeps
    // the snapshot React already rendered, so subscribing doesn't force a second render.
    if (storedRaw() !== cachedRaw) {
      invalidate();
    }
    const onStorage = (event: StorageEvent) => {
      if (event.key === null || event.key === PER_MODEL_CONFIG_STORAGE_KEY) {
        notify();
      }
    };
    window.addEventListener(PER_MODEL_CONFIG_UPDATED_EVENT, notify);
    window.addEventListener("storage", onStorage);
    detach = () => {
      window.removeEventListener(PER_MODEL_CONFIG_UPDATED_EVENT, notify);
      window.removeEventListener("storage", onStorage);
    };
  }
  return () => {
    listeners.delete(onChange);
    if (listeners.size === 0 && detach) {
      detach();
      detach = null;
      invalidate();
    }
  };
}

export function useSavedRunSettings(
  target: ModelPickTarget | null,
): PerModelConfig | null {
  return useSyncExternalStore(
    subscribeSavedRunSettings,
    () => (target ? savedRunSettings(target) : null),
    () => null,
  );
}

export function useHasSavedRunSettings(
  target: ModelPickTarget | null,
): boolean {
  return useSyncExternalStore(
    subscribeSavedRunSettings,
    () => (target ? savedRunSettings(target) !== null : false),
    () => false,
  );
}

// An editor opened after the reset seeded defaults; left alone, its next load forgets again.
function restoreDraftAfterUndo(
  draftKey: string,
  atReset: ModelConfigDraftSnapshot | null,
  snapshot: PerModelConfig,
): void {
  const draft = readModelConfigDraft(draftKey);
  if (!draft || (atReset && !atReset.savedRemember)) {
    return;
  }
  if (
    !atReset &&
    draft.appliedLiveSignature === "none" &&
    !isModelConfigDraftEdited(draftKey)
  ) {
    replaceModelConfigDraft(draftKey, snapshot, {
      remember: true,
      savedRemember: true,
    });
    return;
  }
  setModelConfigDraftRemember(draftKey, true, true);
}

/** Drops a model's saved run settings here and on the server, like the panel's Forget. Returns an
 *  undo that puts them back, or null when there was nothing to forget or the delete failed. */
export function forgetRunSettings(
  target: ModelPickTarget,
): (() => boolean) | null {
  const [id, variant] = settingsKey(target);
  const resolved = resolveInitialConfig(id, variant);
  if (!resolved.remembered) {
    return null;
  }
  const snapshot = resolved.config;
  if (!deletePerModelConfig(id, variant)) {
    return null;
  }
  const mirrored = target.apiLoadable ?? target.isGguf;
  if (mirrored) {
    syncModelOverride(id, variant, null);
  }
  // The loaded model's editor stays mounted in the sidebar, so its draft would keep saying saved.
  const draftKey = modelConfigDraftKey(id, variant);
  const draft = readModelConfigDraft(draftKey);
  if (draft?.savedRemember) {
    setModelConfigDraftRemember(draftKey, false, false);
  }
  return () => {
    if (!savePerModelConfig(id, variant, snapshot)) {
      return false;
    }
    if (mirrored) {
      syncModelOverride(id, variant, snapshot);
    }
    restoreDraftAfterUndo(draftKey, draft ?? null, snapshot);
    return true;
  };
}
