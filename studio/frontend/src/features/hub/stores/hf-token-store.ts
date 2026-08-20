


import { AUTH_SESSION_CLEARED_EVENT } from "../../auth/session-events.ts";
import { reconcileLegacyHfToken } from "../../credentials/reconciliation.ts";

import { create } from "zustand";
type HfTokenApi = typeof import("../api/hf-token-api");

async function loadHfTokenApi(): Promise<HfTokenApi> {
  return import("../api/hf-token-api");
}

const HF_TOKEN_KEY = "unsloth_hf_token";
const LEGACY_TRAINING_KEY = "unsloth_training_config_v1";
const HF_TOKEN_MIGRATION_KEY = "unsloth_hf_token_migration_v1";

const HF_TOKEN_SYNC_KEY = "unsloth_hf_token_backend_revision";

let backendNotificationRevision = 0;

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
    const migrationCopy = window.localStorage.getItem(HF_TOKEN_MIGRATION_KEY);
    if (migrationCopy !== null) return normalizeHfToken(migrationCopy);
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

function removeLegacyToken(expectedToken: string): void {
  const expected = normalizeHfToken(expectedToken);
  if (!expected) return;
  if (normalizeHfToken(stagedLegacyToken) === expected) stagedLegacyToken = "";
  if (!canUseStorage()) return;
  try {
    const direct = window.localStorage.getItem(HF_TOKEN_KEY);
    if (direct !== null && normalizeHfToken(direct) === expected) {
      window.localStorage.removeItem(HF_TOKEN_KEY);
    }
    const migrationCopy = window.localStorage.getItem(HF_TOKEN_MIGRATION_KEY);
    if (migrationCopy !== null && normalizeHfToken(migrationCopy) === expected) {
      window.localStorage.removeItem(HF_TOKEN_MIGRATION_KEY);
    }
    const raw = window.localStorage.getItem(LEGACY_TRAINING_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as { state?: Record<string, unknown> };
    const trainingToken = parsed.state?.hfToken;
    if (
      !parsed.state ||
      typeof trainingToken !== "string" ||
      normalizeHfToken(trainingToken) !== expected
    ) {
      return;
    }
    delete parsed.state.hfToken;
    window.localStorage.setItem(LEGACY_TRAINING_KEY, JSON.stringify(parsed));
  } catch {
    // Keep legacy data untouched on malformed/unavailable storage.
  }
}

let persistenceRevision = 0;
let persistenceChain: Promise<void> = Promise.resolve();
let lastPersistedToken = "";

function announcePersistedTokenChange(): void {
  if (!canUseStorage()) return;
  try {
    window.localStorage.setItem(
      HF_TOKEN_SYNC_KEY,
      `${Date.now()}:${Math.random().toString(36).slice(2)}`,
    );
  } catch {
    // The current tab is already synchronized; cross-tab sync is best effort.
  }
}

function persistenceErrorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Could not save the Hugging Face token.";
}

function persistTokenToBackend(token: string): void {
  const revision = ++persistenceRevision;
  const sessionRevision = authSessionRevision;
  useHfTokenStore.setState({ isPersisting: true, persistenceError: null });
  persistenceChain = persistenceChain
    .catch(() => undefined)
    .then(async () => {
      // Collapse rapid field edits before they reach the network. In-flight
      // writes remain ordered, so an older response can never win last.
      if (revision !== persistenceRevision) return;

      const legacyTokenBeforeSave = loadLegacyToken();
      try {
        const { clearSavedHfToken, loadSavedHfToken, saveHfToken } =
          await loadHfTokenApi();
        const notificationRevisionBeforeWrite = backendNotificationRevision;
        let response = token
          ? await saveHfToken(token)
          : await clearSavedHfToken();

        if (notificationRevisionBeforeWrite !== backendNotificationRevision) {
          // A cross-tab commit landed while this write's response was delayed.
          // Read until no newer notification arrives during the GET, so the
          // value applied below is the backend's latest committed credential.
          let notificationRevision: number;
          do {
            notificationRevision = backendNotificationRevision;
            response = await loadSavedHfToken();
          } while (notificationRevision !== backendNotificationRevision);
        }
        const persistedToken = response.token
          ? normalizeHfToken(response.token)
          : "";
        if (sessionRevision !== authSessionRevision) return;
        lastPersistedToken = persistedToken;
        if (revision === persistenceRevision) {
          removeLegacyToken(legacyTokenBeforeSave);
          announcePersistedTokenChange();
          useHfTokenStore.setState({
            token: persistedToken,
            isPersisting: false,
            persistenceError: null,
          });
        }
      } catch (error) {
        if (revision === persistenceRevision) {
          useHfTokenStore.setState({
            token: lastPersistedToken,
            isPersisting: false,
            persistenceError: persistenceErrorMessage(error),
          });
        }
      }
    });
}


/** Wait until the latest queued token edit is durable, or surface its save error. */
export async function waitForHfTokenPersistence(): Promise<void> {
  while (true) {
    const revision = persistenceRevision;
    await persistenceChain;
    if (revision !== persistenceRevision) continue;
    const state = useHfTokenStore.getState();
    if (state.isPersisting) continue;
    if (state.persistenceError) throw new Error(state.persistenceError);
    return;
  }
}


interface HfTokenStore {
  token: string;
  isPersisting: boolean;
  persistenceError: string | null;
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

    isPersisting: false,
    persistenceError: null,
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
  if (canUseStorage()) {
    try {
      window.localStorage.setItem(HF_TOKEN_MIGRATION_KEY, token);
    } catch {
      // The in-memory copy still keeps this session working.
    }
  }
  if (!useHfTokenStore.getState().token) {
    useHfTokenStore.setState({ token });
  }
}



let hydrationPromise: Promise<void> | null = null;
let hydrationReplayRequested = false;

/** Server-first, retry-safe migration and hydration after authentication. */
export function hydrateHfTokenFromBackend(): Promise<void> {
  if (hydrationPromise) return hydrationPromise;
  const sessionRevision = authSessionRevision;
  const persistenceRevisionAtStart = persistenceRevision;
  const persistenceWasPending = useHfTokenStore.getState().isPersisting;
  const canApplyHydratedToken = () =>
    !persistenceWasPending && persistenceRevisionAtStart === persistenceRevision;
  const assertCurrentSession = () => {
    if (sessionRevision !== authSessionRevision) {
      throw new Error("Authentication session changed during credential hydration.");
    }
  };

  hydrationPromise = (async () => {
    const { loadSavedHfToken, migrateHfToken } = await loadHfTokenApi();
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
        const response = await migrateHfToken(token);
        assertCurrentSession();
        return response;
      },
      applyToken: (token) => {
        assertCurrentSession();
        if (!canApplyHydratedToken()) return;
        lastPersistedToken = normalizeHfToken(token);
        useHfTokenStore.setState({
          token: lastPersistedToken,
          isPersisting: false,
          persistenceError: null,
        });
      },
      removeLegacyToken: (expectedToken) => {
        assertCurrentSession();
        removeLegacyToken(expectedToken);
      },
    });
  })().finally(() => {
    if (sessionRevision === authSessionRevision) hydrationPromise = null;
  });
  return hydrationPromise;
}


/** Re-read after an active hydration when another tab reports a newer backend value. */
export async function refreshHfTokenFromBackend(): Promise<void> {

  backendNotificationRevision += 1;
  if (!hydrationPromise) return hydrateHfTokenFromBackend();
  hydrationReplayRequested = true;
  try {
    await hydrationPromise;
  } catch {
    // The replay is the authoritative retry for the cross-tab notification.
  }
  if (!hydrationReplayRequested) return;
  hydrationReplayRequested = false;
  return hydrateHfTokenFromBackend();
}

function resetHfCredentialSession(): void {
  authSessionRevision += 1;
  persistenceRevision += 1;
  hydrationPromise = null;
  hydrationReplayRequested = false;
  serverCredentialHydrated = false;

  stagedLegacyToken = "";
  lastPersistedToken = "";
  useHfTokenStore.setState({
    token: "",
    isPersisting: false,
    persistenceError: null,
  });
}

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, resetHfCredentialSession);

  window.addEventListener("storage", (event) => {
    if (event.key !== HF_TOKEN_SYNC_KEY || !event.newValue) return;
    void refreshHfTokenFromBackend().catch(() => undefined);
  });
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
