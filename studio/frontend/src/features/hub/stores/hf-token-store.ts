// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AUTH_SESSION_CLEARED_EVENT } from "../../auth/session-events.ts";
import { reconcileLegacyHfToken } from "../../credentials/reconciliation.ts";

import { create } from "zustand";
type HfTokenApi = typeof import("../api/hf-token-api");

async function loadHfTokenApi(): Promise<HfTokenApi> {
  return import("../api/hf-token-api");
}

const HF_TOKEN_KEY = "unsloth_hf_token";
const LEGACY_TRAINING_KEY = "unsloth_training_config_v1";

let stagedLegacyToken = "";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

export function normalizeHfToken(raw: string): string {
  return raw.replace(/^[\s"']+|[\s"']+$/g, "");
}

function loadLegacyToken(): string {
  if (!canUseStorage()) return stagedLegacyToken;
  try {
    const direct = window.localStorage.getItem(HF_TOKEN_KEY);
    if (direct !== null) return normalizeHfToken(direct);
    const legacy = window.localStorage.getItem(LEGACY_TRAINING_KEY);
    if (!legacy) return stagedLegacyToken;
    const parsed = JSON.parse(legacy) as { state?: Record<string, unknown> };
    const token = parsed?.state?.hfToken;
    return typeof token === "string"
      ? normalizeHfToken(token)
      : stagedLegacyToken;
  } catch {
    return stagedLegacyToken;
  }
}

function removeLegacyToken(): void {

  stagedLegacyToken = "";
  if (!canUseStorage()) return;
  try {
    window.localStorage.removeItem(HF_TOKEN_KEY);
    const raw = window.localStorage.getItem(LEGACY_TRAINING_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as { state?: Record<string, unknown> };
    if (!parsed.state || !("hfToken" in parsed.state)) return;
    delete parsed.state.hfToken;
    window.localStorage.setItem(LEGACY_TRAINING_KEY, JSON.stringify(parsed));
  } catch {
    // Keep legacy data untouched on malformed/unavailable storage.
  }
}

let persistenceRevision = 0;
let persistenceChain: Promise<void> = Promise.resolve();


function persistTokenToBackend(token: string): void {
  const revision = ++persistenceRevision;
  persistenceChain = persistenceChain
    .catch(() => undefined)
    .then(async () => {
      // Collapse rapid field edits before they reach the network. In-flight
      // writes remain ordered, so an older response can never win last.
      if (revision !== persistenceRevision) return;
      const { clearSavedHfToken, saveHfToken } = await loadHfTokenApi();
      if (token) {
        await saveHfToken(token);
      } else {
        await clearSavedHfToken();
      }
      if (revision === persistenceRevision) removeLegacyToken();
    })
    .catch(() => undefined);
}


interface HfTokenStore {
  token: string;
  setToken: (value: string) => void;
  clearToken: () => void;
}

export const useHfTokenStore = create<HfTokenStore>((set) => {
  const applyNormalizedToken = (value: string) => {
    const token = normalizeHfToken(value);
    set((state) => (state.token === token ? state : { token }));
    return token;
  };
  return {
    // Legacy value remains available until authenticated bootstrap reconciles it.
    token: loadLegacyToken(),
    setToken: (value) => {
      persistTokenToBackend(applyNormalizedToken(value));
    },
    clearToken: () => {
      applyNormalizedToken("");
      persistTokenToBackend("");
    },
  };
});

let serverCredentialHydrated = false;

let authSessionRevision = 0;

/** Retain a pre-v12 training-store token as migration input until server save succeeds. */
export function stageLegacyHfTokenForMigration(value: string): void {
  if (serverCredentialHydrated) return;
  const token = normalizeHfToken(value);
  if (!token) return;
  stagedLegacyToken ||= token;
  if (!useHfTokenStore.getState().token) {
    useHfTokenStore.setState({ token });
  }
}

let hydrationPromise: Promise<void> | null = null;

/** Server-first, retry-safe migration and hydration after authentication. */
export function hydrateHfTokenFromBackend(): Promise<void> {
  if (hydrationPromise) return hydrationPromise;
  const sessionRevision = authSessionRevision;
  const assertCurrentSession = () => {
    if (sessionRevision !== authSessionRevision) {
      throw new Error("Authentication session changed during credential hydration.");
    }
  };

  hydrationPromise = (async () => {
    const { loadSavedHfToken, saveHfToken } = await loadHfTokenApi();
    await reconcileLegacyHfToken({
      loadSavedToken: async () => {
        const response = await loadSavedHfToken();
        assertCurrentSession();
        serverCredentialHydrated = true;
        return response;
      },
      getLegacyToken: () => {
        assertCurrentSession();
        return loadLegacyToken();
      },
      saveLegacyToken: async (token) => {
        const response = await saveHfToken(token);
        assertCurrentSession();
        return response;
      },
      applyToken: (token) => {
        assertCurrentSession();
        useHfTokenStore.setState({ token: normalizeHfToken(token) });
      },
      removeLegacyToken: () => {
        assertCurrentSession();
        removeLegacyToken();
      },
    });
  })().finally(() => {
    if (sessionRevision === authSessionRevision) hydrationPromise = null;
  });
  return hydrationPromise;
}

function resetHfCredentialSession(): void {
  authSessionRevision += 1;
  persistenceRevision += 1;
  hydrationPromise = null;
  serverCredentialHydrated = false;
  stagedLegacyToken = "";
  useHfTokenStore.setState({ token: "" });
}

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, resetHfCredentialSession);
}

export function getHfToken(): string {
  return useHfTokenStore.getState().token;
}

// Keep a plain zustand store's `hfToken` in sync with the shared token.
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

// HF's JS client throws on a non-empty token without the `hf_` prefix.
export function hfApiToken(
  token: string | undefined | null,
): string | undefined {
  const normalized = token ? normalizeHfToken(token) : "";
  return normalized.startsWith("hf_") ? normalized : undefined;
}
