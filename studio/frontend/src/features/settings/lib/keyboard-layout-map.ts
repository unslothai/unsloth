// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

/** `KeyboardEvent.code` → the key's label on the active layout (from the Keyboard API). */
export type KeyboardLayoutMapLike = ReadonlyMap<string, string>;

let cached: Map<string, string> | null | undefined;
let loadPromise: Promise<Map<string, string> | null> | null = null;
const listeners = new Set<() => void>();

type KeyboardLayoutMapSource = {
  getLayoutMap(): Promise<ReadonlyMap<string, string>>;
};

function keyboardLayoutMapSource(): KeyboardLayoutMapSource | null {
  if (typeof navigator === "undefined") return null;
  const keyboard = (navigator as { keyboard?: KeyboardLayoutMapSource }).keyboard;
  return keyboard?.getLayoutMap ? keyboard : null;
}

function notify() {
  for (const listener of [...listeners]) listener();
}

/** Cached layout map, or `undefined` while the first load is in flight. */
export function getCachedKeyboardLayoutMap():
  | Map<string, string>
  | null
  | undefined {
  return cached;
}

/** Load once per document; returns null when the API is missing or blocked. */
export function loadKeyboardLayoutMap(): Promise<Map<string, string> | null> {
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    const keyboard = keyboardLayoutMapSource();
    if (!keyboard) {
      cached = null;
      notify();
      return null;
    }
    try {
      const map = await keyboard.getLayoutMap();
      cached = new Map(map);
    } catch {
      cached = null;
    }
    notify();
    return cached;
  })();
  return loadPromise;
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  void loadKeyboardLayoutMap();
  return () => listeners.delete(listener);
}

function getSnapshot() {
  return cached;
}

function getServerSnapshot() {
  return undefined;
}

/** Active keyboard layout for shortcut labels; undefined until resolved. */
export function useKeyboardLayoutMap():
  | Map<string, string>
  | null
  | undefined {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
