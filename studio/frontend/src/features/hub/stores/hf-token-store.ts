// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { bumpInventoryVersion } from "./inventory-events";

const HF_TOKEN_KEY = "unsloth_hf_token";
const LEGACY_TRAINING_KEY = "unsloth_training_config_v1";
const AUTH_SESSION_CHANGED_EVENT = "unsloth-auth-session-changed";
let storageSyncStarted = false;
let storageSyncListener: ((event: StorageEvent) => void) | null = null;
let authSessionListener: (() => void) | null = null;
let notebookHydration: Promise<void> | null = null;

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function loadInitial(): string {
  if (!canUseStorage()) return "";
  try {
    const direct = window.localStorage.getItem(HF_TOKEN_KEY);
    if (direct !== null) {
      const normalized = normalize(direct);
      if (normalized !== direct) persist(normalized);
      return normalized;
    }
    const legacy = window.localStorage.getItem(LEGACY_TRAINING_KEY);
    if (legacy) {
      const parsed = JSON.parse(legacy) as {
        state?: Record<string, unknown>;
        [key: string]: unknown;
      };
      const fromTraining = parsed?.state?.hfToken;
      if (typeof fromTraining === "string" && fromTraining.length > 0) {
        const normalized = normalize(fromTraining);
        // Copy only: training-config-store reads its own hfToken; deleting the legacy field drops it.
        persist(normalized);
        return normalized;
      }
    }
  } catch {
    return "";
  }
  return "";
}

function persist(value: string): void {
  if (!canUseStorage()) return;
  try {
    window.localStorage.setItem(HF_TOKEN_KEY, value);
  } catch {
    void 0;
  }
}

function normalize(raw: string): string {
  return raw.replace(/^[\s"']+|[\s"']+$/g, "");
}

function stopStorageSync(): void {
  if (!canUseStorage()) return;
  if (storageSyncListener !== null) {
    window.removeEventListener("storage", storageSyncListener);
    storageSyncListener = null;
  }
  if (authSessionListener !== null) {
    window.removeEventListener(AUTH_SESSION_CHANGED_EVENT, authSessionListener);
    authSessionListener = null;
  }
  storageSyncStarted = false;
}

if (import.meta.hot) {
  import.meta.hot.dispose(stopStorageSync);
}

interface HfTokenStore {
  token: string;
  setToken: (value: string) => void;
  clearToken: () => void;
}

export const useHfTokenStore = create<HfTokenStore>((set) => {
  const applyToken = (value: string, shouldPersist: boolean) => {
    const next = normalize(value);
    if (shouldPersist || next !== value) persist(next);
    let changed = false;
    set((state) => {
      if (state.token === next) return state;
      changed = true;
      return { token: next };
    });
    if (changed) bumpInventoryVersion();
  };

  if (!storageSyncStarted && canUseStorage()) {
    storageSyncStarted = true;
    storageSyncListener = (event) => {
      if (event.key !== HF_TOKEN_KEY) return;
      applyToken(event.newValue ?? "", false);
    };
    window.addEventListener("storage", storageSyncListener);
  }

  return {
    token: loadInitial(),
    setToken: (value) => applyToken(value, true),
    clearToken: () => applyToken("", true),
  };
});

// Colab and Kaggle secrets live in the notebook kernel and cannot be read by
// browser JavaScript. Ask the authenticated backend to resolve HF_TOKEN, then
// hydrate the shared store so every token field and operation sees it.
function hydrateNotebookHfToken(): Promise<void> {
  if (useHfTokenStore.getState().token) {
    return Promise.resolve();
  }
  if (notebookHydration) {
    return notebookHydration;
  }
  notebookHydration = import("../api/notebook-token")
    .then(({ loadNotebookHfToken }) => loadNotebookHfToken())
    .then((token) => {
      if (token && !useHfTokenStore.getState().token) {
        useHfTokenStore.getState().setToken(token);
      }
    })
    .catch(() => undefined)
    .finally(() => {
      notebookHydration = null;
    });
  return notebookHydration;
}

if (canUseStorage() && !useHfTokenStore.getState().token) {
  // The first request handles returning sessions. A fresh login dispatches the
  // event below, retrying after credentials exist instead of losing the token
  // to an early 401 during application module initialization.
  authSessionListener = () => {
    hydrateNotebookHfToken().catch(() => undefined);
  };
  window.addEventListener(AUTH_SESSION_CHANGED_EVENT, authSessionListener);
  hydrateNotebookHfToken().catch(() => undefined);
}

export function getHfToken(): string {
  return useHfTokenStore.getState().token;
}

// Keep a plain zustand store's `hfToken` field in sync with the shared token:
// seed the current value, then mirror later edits. Returns the unsubscribe so
// callers can wire it to HMR disposal.
export function mirrorHfTokenInto<T extends { hfToken: string }>(store: {
  getState: () => T;
  setState: (partial: Partial<T>) => void;
}): () => void {
  store.setState({ hfToken: getHfToken() } as Partial<T>);
  return useHfTokenStore.subscribe((state) => {
    if (store.getState().hfToken !== state.token) {
      store.setState({ hfToken: state.token } as Partial<T>);
    }
  });
}

// HF's JS client throws on a non-empty token that isn't `hf_...` instead of
// browsing anonymously, so treat anything malformed as no token.
export function hfApiToken(
  token: string | undefined | null,
): string | undefined {
  return token?.startsWith("hf_") ? token : undefined;
}
