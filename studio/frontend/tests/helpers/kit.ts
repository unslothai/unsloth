// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { register } from "node:module";

import type { ResidentAdoptionState } from "../../src/features/hub/lib/adopt-inference-status.ts";
import type { ResidentStatusRefreshTargets } from "../../src/features/hub/lib/resident-status-refresh.ts";

/** A `src` module read as text, path relative to `src`. */
export function readSrc(relative: string): string {
  return readFileSync(new URL(`../../src/${relative}`, import.meta.url), "utf8");
}

const UI_SPACE_SCALE =
  /calc\(\s*(-?[\d.]+(?:px|rem|em))\s*\*\s*var\(--ui-space-scale\s*,\s*1\)\s*\)/g;

/** Source with --ui-space-scale collapsed to its default-font-size length. */
export function atDefaultUiScale(source: string): string {
  return source.replace(UI_SPACE_SCALE, "$1");
}

/** A repo file read as text, relative to `studio/frontend/tests`. */
export function readText(relative: string): string {
  return readFileSync(new URL(`../${relative}`, import.meta.url), "utf8");
}

export function readSrcAsync(relative: string): Promise<string> {
  return readFile(new URL(`../../src/${relative}`, import.meta.url), "utf8");
}

/** Register vite/tsconfig "bundler" resolution. Call before importing such src modules. */
export function registerBundlerResolver(): void {
  register("../bundler-resolver.mjs", import.meta.url);
}

export function registerStoreStubResolver(): void {
  register("../store-stub-resolver.mjs", import.meta.url);
}

export type StorageFake = {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  removeItem: (key: string) => void;
};

/** In-memory localStorage plus a window whose listeners `fireWindowEvent` can drive. */
export function installLocalStorageFake(): {
  store: Map<string, string>;
  storage: StorageFake & Pick<Storage, "length" | "key">;
  fireWindowEvent: (type: string, event: unknown) => number;
} {
  const store = new Map<string, string>();
  const storage: StorageFake & Pick<Storage, "length" | "key"> = {
    get length() { return store.size; },
    key: (index) => [...store.keys()][index] ?? null,
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
    // lib/api-base reads location.protocol; cross-tab stores subscribe to "storage".
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
    // Code guarded on `typeof window` also reaches for document.
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
    fireWindowEvent: (type: string, event: unknown) => {
      const set = listeners.get(type);
      for (const fn of set ?? []) {
        fn(event);
      }
      return set?.size ?? 0;
    },
  };
}

export function emptyStore(
  overrides: Partial<ResidentAdoptionState> = {},
): ResidentAdoptionState {
  return {
    checkpoint: null,
    checkpointIsExternal: false,
    activeGgufVariant: null,
    modelLoading: false,
    idleUnloadArmed: false,
    ...overrides,
  };
}

export function spies() {
  const calls: string[] = [];
  const previouslySeen: { checkpoint: string | null; ggufVariant: string | null }[] =
    [];
  return {
    calls,
    previouslySeen,
    actions: {
      setCheckpoint(checkpointId: string, ggufVariant: string | null) {
        calls.push(`setCheckpoint:${checkpointId}:${ggufVariant ?? ""}`);
      },
      applyStatus(previous: {
        checkpoint: string | null;
        ggufVariant: string | null;
      }) {
        calls.push("applyStatus");
        previouslySeen.push(previous);
      },
    },
  };
}

export function fakeTargets(): ResidentStatusRefreshTargets & {
  hidden: boolean;
  fire: (target: "window" | "document", type: string) => void;
  listenerCount: () => number;
} {
  const listeners = new Map<string, Set<EventListenerOrEventListenerObject>>();
  const key = (target: string, type: string) => `${target}:${type}`;
  const make = (target: "window" | "document") => ({
    addEventListener(type: string, fn: EventListenerOrEventListenerObject) {
      const set = listeners.get(key(target, type)) ?? new Set();
      set.add(fn);
      listeners.set(key(target, type), set);
    },
    removeEventListener(type: string, fn: EventListenerOrEventListenerObject) {
      listeners.get(key(target, type))?.delete(fn);
    },
  });
  const visibility = { hidden: false };
  const state = {
    get hidden() {
      return visibility.hidden;
    },
    set hidden(next: boolean) {
      visibility.hidden = next;
    },
    window: make("window"),
    document: {
      ...make("document"),
      get hidden() {
        return visibility.hidden;
      },
    },
    fire(target: "window" | "document", type: string) {
      for (const fn of listeners.get(key(target, type)) ?? []) {
        (fn as EventListener)(new Event(type));
      }
    },
    listenerCount() {
      let total = 0;
      for (const set of listeners.values()) total += set.size;
      return total;
    },
  };
  return state as never;
}
