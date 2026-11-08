// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

const SRC = fileURLToPath(new URL("../../src/", import.meta.url));

const ZUSTAND_STUB =
  "data:text/javascript," +
  encodeURIComponent(`
    export const create = (fn) => {
      let state;
      const listeners = new Set();
      const set = (updater) => {
        state = typeof updater === "function" ? { ...state, ...updater(state) } : { ...state, ...updater };
        listeners.forEach((l) => l(state));
      };
      const get = () => state;
      state = fn(set, get, { getState: get, setState: set, subscribe: (l) => { listeners.add(l); return () => listeners.delete(l); } });
      const useStore = (selector) => (selector ? selector(state) : state);
      useStore.getState = get;
      useStore.setState = set;
      useStore.subscribe = (l) => { listeners.add(l); return () => listeners.delete(l); };
      return useStore;
    };
  `);

const AUTH_STUB =
  "data:text/javascript," +
  encodeURIComponent(`
    export const authFetch = async () => ({ ok: true, json: async () => ({}) });
  `);

function firstExisting(base) {
  for (const candidate of [`${base}.ts`, `${base}/index.ts`, base]) {
    if (existsSync(candidate)) {
      return pathToFileURL(candidate).href;
    }
  }
  return null;
}

export function resolve(specifier, context, next) {
  if (specifier === "zustand") {
    return { url: ZUSTAND_STUB, shortCircuit: true };
  }
  if (specifier === "@/features/auth") {
    return { url: AUTH_STUB, shortCircuit: true };
  }
  if (specifier.startsWith("@/")) {
    const resolved = firstExisting(SRC + specifier.slice(2));
    if (resolved) return next(resolved, context);
  }
  if (specifier.startsWith(".") && context.parentURL?.startsWith("file:")) {
    const resolved = firstExisting(fileURLToPath(new URL(specifier, context.parentURL)));
    if (resolved) return next(resolved, context);
  }
  return next(specifier, context);
}
