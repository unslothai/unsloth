// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

// Labels only: requests always send the id.
const MODEL_DISPLAY_NAMES_KEY = "unsloth_chat_provider_model_names";
const NAMES = new Map<string, Record<string, string>>();
let hydrated = false;
let version = 0;
const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useModelDisplayNamesVersion(): number {
  return useSyncExternalStore(subscribe, () => version, () => version);
}

function hydrate(): void {
  if (hydrated) return;
  hydrated = true;
  if (typeof window === "undefined") return;
  try {
    const parsed: unknown = JSON.parse(
      localStorage.getItem(MODEL_DISPLAY_NAMES_KEY) ?? "{}",
    );
    if (!parsed || typeof parsed !== "object") return;
    for (const [providerType, names] of Object.entries(parsed)) {
      if (!names || typeof names !== "object") continue;
      NAMES.set(
        providerType,
        Object.fromEntries(
          Object.entries(names).filter(
            (entry): entry is [string, string] => typeof entry[1] === "string",
          ),
        ),
      );
    }
  } catch {
    // Relearned on the next listing.
  }
}

export function learnModelDisplayNames(
  providerType: string,
  models: readonly { id: string; display_name?: string | null }[],
): void {
  if (!providerType) return;
  hydrate();
  const names = { ...NAMES.get(providerType) };
  let changed = false;
  for (const model of models) {
    const id = model.id?.trim();
    const name = model.display_name?.trim();
    if (!id) continue;
    if (name && name !== id) {
      if (names[id] !== name) {
        names[id] = name;
        changed = true;
      }
    } else if (id in names) {
      delete names[id];
      changed = true;
    }
  }
  if (!changed) return;
  NAMES.set(providerType, names);
  version += 1;
  for (const listener of listeners) listener();
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(
      MODEL_DISPLAY_NAMES_KEY,
      JSON.stringify(Object.fromEntries(NAMES)),
    );
  } catch {
    // Kept for this session.
  }
}

export function providerModelDisplayName(
  providerType: string | null | undefined,
  modelId: string,
): string {
  if (!providerType) return modelId;
  hydrate();
  return NAMES.get(providerType)?.[modelId] ?? modelId;
}
