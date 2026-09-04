// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// These stores are used outside React and are not part of their features' public barrels.
// eslint-disable-next-line no-restricted-imports
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth/session-events";
// eslint-disable-next-line no-restricted-imports
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
// eslint-disable-next-line no-restricted-imports
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { type HfTokenValidationResult, validateHfToken } from "./api";
import { useHfTokenWarningStore } from "./store";

export interface PreparedHfToken {
  proceed: boolean;
  token: string | null;
}

interface PrepareHfTokenOptions {
  allowAnonymous?: boolean;
}

// A caller can retain the pre-dialog payload while the shared store is cleared. Remember that
// one-session choice so a follow-up /load does not prompt again after an anonymous /validate.
const anonymousForSession = new Set<string>();

// One load prepares the same token three times (pollers, validateModel, loadModel), so a
// burst collapses to one call rather than threading a prepared credential through every
// signature. Short and positive-only: "invalid" is never cached, so the dialog still
// appears and an expiring token is re-checked once the window lapses.
const VALIDATION_REUSE_MS = 15_000;
// Bumped on every clear, so an in-flight request cannot repopulate the cache afterwards.
let cacheGeneration = 0;
const recentlyValid = new Map<string, number>();
const inFlight = new Map<string, Promise<HfTokenValidationResult>>();

// Entries hold the raw bearer token, so they go as soon as they stop being useful.
function dropExpiredValidations(now: number): void {
  for (const [cached, validAt] of recentlyValid) {
    if (now - validAt >= VALIDATION_REUSE_MS) {
      recentlyValid.delete(cached);
    }
  }
}

function validateOncePerBurst(token: string): Promise<HfTokenValidationResult> {
  const now = Date.now();
  dropExpiredValidations(now);
  const validAt = recentlyValid.get(token);
  if (validAt != null && now - validAt < VALIDATION_REUSE_MS) {
    return Promise.resolve({ status: "valid", retryAfterSeconds: null });
  }
  const pending = inFlight.get(token);
  if (pending) {
    return pending;
  }
  // Clearing cannot cancel an in-flight request, so a logout mid-validation would have
  // the late resolution write the raw token back, and expiry only runs on the next
  // preparation. The generation makes a stale resolution non-cacheable.
  const generation = cacheGeneration;
  const request = validateHfToken(token)
    .then((result) => {
      // Only a definitive pass is reusable: "unavailable" says nothing, and reusing it
      // would suppress the dialog for a token that is genuinely bad.
      if (result.status === "valid" && generation === cacheGeneration) {
        recentlyValid.set(token, Date.now());
      }
      return result;
    })
    .finally(() => {
      // Only if still ours: a forget-then-prepare can put a live replacement in this
      // slot, and an unconditional delete would evict it.
      if (inFlight.get(token) === request) {
        inFlight.delete(token);
      }
    });
  inFlight.set(token, request);
  return request;
}

// Called with no argument the whole cache goes; with one, just that credential.
export function forgetHfTokenValidation(token?: string): void {
  cacheGeneration += 1;
  if (token == null) {
    recentlyValid.clear();
    inFlight.clear();
    return;
  }
  recentlyValid.delete(token);
  inFlight.delete(token);
}

// A logout must not leave a bearer token in module memory, the same reason
// hf-token-store.ts resets on this event. Replacing it drops the superseded key too.
if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    forgetHfTokenValidation();
    // anonymousForSession is deliberately left alone: it predates this cache and records a
    // user's choice rather than a Hub verdict.
  });
}

let lastKnownStoredToken: string | null = null;
let tokenChangeSubscribed = false;

// Subscribed on first use, not at module scope: reading a barrel-imported value while
// this module loads throws if the import cycle re-enters (module-scope-cycle-safety).
function ensureTokenChangeSubscription(): void {
  if (tokenChangeSubscribed || typeof useHfTokenStore.subscribe !== "function") {
    return;
  }
  tokenChangeSubscribed = true;
  // Seeded, not left null: zustand does not fire subscribe on install, so replacing an
  // already-stored token would read as initialization and keep its window.
  lastKnownStoredToken = useHfTokenStore.getState().token?.trim() ?? "";
  useHfTokenStore.subscribe((state) => {
    const next = state.token?.trim() ?? "";
    if (lastKnownStoredToken != null && lastKnownStoredToken !== next) {
      forgetHfTokenValidation(lastKnownStoredToken);
    }
    lastKnownStoredToken = next;
  });
}

export async function prepareHfTokenForUse(
  token: string | null | undefined,
  options: PrepareHfTokenOptions = {},
): Promise<PreparedHfToken> {
  ensureTokenChangeSubscription();
  const normalized = token?.trim() ?? "";
  if (!normalized) {
    return { proceed: true, token: null };
  }
  const allowAnonymous = options.allowAnonymous ?? true;
  if (allowAnonymous && anonymousForSession.has(normalized)) {
    return { proceed: true, token: null };
  }

  let validation: HfTokenValidationResult;
  try {
    validation = await validateOncePerBurst(normalized);
  } catch {
    // Validation is advisory. Let the real operation retain its own error.
    return { proceed: true, token: normalized };
  }
  if (validation.status !== "invalid") {
    // A connectivity failure or rate limit cannot prove a token is bad; let the real operation run.
    return { proceed: true, token: normalized };
  }

  const decision = await useHfTokenWarningStore
    .getState()
    .requestDecision(allowAnonymous);
  if (decision === "anonymous") {
    anonymousForSession.add(normalized);
    const tokenStore = useHfTokenStore.getState();
    if (tokenStore.token === normalized) {
      tokenStore.clearToken();
    }
    forgetHfTokenValidation(normalized);
    return { proceed: true, token: null };
  }
  if (decision === "replace") {
    useSettingsDialogStore.getState().openDialog("general");
  }
  return { proceed: false, token: normalized };
}
