// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";
import { syncModelOverride } from "../api/model-overrides";
import type { ModelPickTarget } from "../components/model-selector/types";
import {
  modelConfigDraftKey,
  readModelConfigDraft,
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

function invalidate(): void {
  savedByKey = new Map();
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
  // Only trusted while subscribed: nothing invalidates the cache otherwise.
  const cached = detach ? savedByKey.get(key) : undefined;
  if (cached !== undefined) {
    return cached;
  }
  const resolved = resolveInitialConfig(id, variant);
  const value = resolved.remembered ? resolved.config : null;
  if (detach) {
    savedByKey.set(key, value);
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
    // Writes made while nobody listened never invalidated the cache.
    invalidate();
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

/** The saved record itself, re-read when any model's saved settings change. */
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
    if (draft?.savedRemember) {
      setModelConfigDraftRemember(draftKey, true, true);
    }
    return true;
  };
}
