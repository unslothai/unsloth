// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { register } from "node:module";

/**
 * Teach the loader the two resolution rules vite and tsconfig's "bundler" mode give the
 * app. Call before dynamically importing any src module that resolves that way.
 */
export function registerBundlerResolver(): void {
  register("../bundler-resolver.mjs", import.meta.url);
}

/** Register the bundler resolver with store dependency stubs. */
export function registerStoreStubResolver(): void {
  register("../store-stub-resolver.mjs", import.meta.url);
}

export type StorageFake = {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  removeItem: (key: string) => void;
};

/**
 * An in-memory localStorage, installed on globalThis under both names the app reads it
 * by. The returned map is the backing store, so a test can stage records before import.
 *
 * The window it installs keeps the listeners registered against it, so `fireWindowEvent`
 * can deliver a cross-tab "storage" event to a store that subscribed on construction.
 */
export function installLocalStorageFake(): {
  store: Map<string, string>;
  storage: StorageFake;
  fireWindowEvent: (type: string, event: unknown) => number;
} {
  const store = new Map<string, string>();
  const storage: StorageFake = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
  };
  const listeners = new Map<string, Set<(event: unknown) => void>>();
  Object.assign(globalThis, {
    // A location too: lib/api-base reads its protocol, pulled in transitively.
    // Stores that sync across tabs subscribe to "storage" on construction.
    window: {
      localStorage: storage,
      location: { protocol: "http:" },
      addEventListener: (type: string, fn: (event: unknown) => void) => {
        const set = listeners.get(type) ?? new Set<(event: unknown) => void>();
        set.add(fn);
        listeners.set(type, set);
      },
      removeEventListener: (type: string, fn: (event: unknown) => void) => {
        listeners.get(type)?.delete(fn);
      },
    },
    // A window implies a document: code guarded on `typeof window` reaches for
    // one, and a fake without it is a browser no browser ever is.
    document: {
      visibilityState: "visible",
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
    },
    localStorage: storage,
  });
  return {
    store,
    storage,
    /** Deliver `event` to every window listener for `type`. Returns how many ran. */
    fireWindowEvent: (type: string, event: unknown) => {
      const set = listeners.get(type);
      for (const fn of set ?? []) {
        fn(event);
      }
      return set?.size ?? 0;
    },
  };
}
