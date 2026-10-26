// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

// Remember the anonymous choice so a later /load does not prompt again.
const anonymousForSession = new Set<string>();

// One load validates the same token three times; cache positive results only, briefly.
const VALIDATION_REUSE_MS = 15_000;
// Bumped on every clear, so an in-flight request cannot repopulate the cache.
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
  const generation = cacheGeneration;
  const request = validateHfToken(token)
    .then((result) => {
      // Only a definitive pass is reusable; "unavailable" would suppress the dialog wrongly.
      if (result.status === "valid" && generation === cacheGeneration) {
        recentlyValid.set(token, Date.now());
      }
      return result;
    })
    .finally(() => {
      // Only if still ours: a replacement may now occupy this slot.
      if (inFlight.get(token) === request) {
        inFlight.delete(token);
      }
    });
  inFlight.set(token, request);
  return request;
}

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

// A logout must not leave a bearer token in module memory.
if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    forgetHfTokenValidation();
    // anonymousForSession records a user choice, not a Hub verdict, so it is kept.
  });
}

let lastKnownStoredToken: string | null = null;
let tokenChangeSubscribed = false;

// Subscribed lazily: module-scope reads throw if the import cycle re-enters.
function ensureTokenChangeSubscription(): void {
  if (tokenChangeSubscribed || typeof useHfTokenStore.subscribe !== "function") {
    return;
  }
  tokenChangeSubscribed = true;
  // Seeded: zustand does not fire subscribe on install.
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
    return { proceed: true, token: normalized };
  }
  if (validation.status !== "invalid") {
    // A connectivity failure or rate limit cannot prove a token is bad.
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
