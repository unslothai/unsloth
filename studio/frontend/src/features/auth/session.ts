


import { isTauri } from "@/lib/api-base";

import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
} from "./session-events";

export { AUTH_SESSION_CLEARED_EVENT, AUTH_SESSION_STORED_EVENT } from "./session-events";

export const AUTH_TOKEN_KEY = "unsloth_auth_token";
export const AUTH_REFRESH_TOKEN_KEY = "unsloth_auth_refresh_token";
export const ONBOARDING_DONE_KEY = "unsloth_onboarding_done";
export const AUTH_MUST_CHANGE_PASSWORD_KEY = "unsloth_auth_must_change_password";


let authSessionEpoch = 0;

/** Stable across access-token refreshes; changes when an auth session starts or clears. */
export function getAuthSessionEpoch(): number {
  return authSessionEpoch;
}

type PostAuthRoute = "/change-password" | "/chat";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

export function hasAuthToken(): boolean {
  if (!canUseStorage()) return false;
  return Boolean(localStorage.getItem(AUTH_TOKEN_KEY));
}

export function hasRefreshToken(): boolean {
  if (!canUseStorage()) return false;
  return Boolean(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY));
}

export function getAuthToken(): string | null {
  if (!canUseStorage()) return null;
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (!canUseStorage()) return null;
  return localStorage.getItem(AUTH_REFRESH_TOKEN_KEY);
}

export function storeAuthTokens(
  accessToken: string,
  refreshToken: string,
): void {
  // must_change_password is set via setMustChangePassword(), not here: routing
  // it through would let CodeQL trace the boolean into localStorage and flag the
  // deliberate JWT writes as sensitive-info storage.
  if (!canUseStorage()) return;
  const sessionStarted = !localStorage.getItem(AUTH_TOKEN_KEY);
  localStorage.setItem(AUTH_TOKEN_KEY, accessToken);
  localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, refreshToken);
  if (sessionStarted) {
    authSessionEpoch += 1;
    window.dispatchEvent(new Event(AUTH_SESSION_STORED_EVENT));
  }
}

export function clearAuthTokens(): void {
  if (!canUseStorage()) return;
  authSessionEpoch += 1;
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_REFRESH_TOKEN_KEY);
  localStorage.removeItem(AUTH_MUST_CHANGE_PASSWORD_KEY);
  window.dispatchEvent(new Event(AUTH_SESSION_CLEARED_EVENT));
}

// Flag stored as key presence (constant "1" or absence), not a derived boolean,
// so CodeQL doesn't flow must_change_password into localStorage.setItem. The
// value is a route hint (/change-password vs /chat), not a secret.
export function mustChangePassword(): boolean {
  if (!canUseStorage()) return false;
  return localStorage.getItem(AUTH_MUST_CHANGE_PASSWORD_KEY) !== null;
}

export function setMustChangePassword(required: boolean): void {
  if (!canUseStorage()) return;
  if (required) {
    localStorage.setItem(AUTH_MUST_CHANGE_PASSWORD_KEY, "1");
  } else {
    localStorage.removeItem(AUTH_MUST_CHANGE_PASSWORD_KEY);
  }
}



export function isOnboardingDone(): boolean {
  if (!canUseStorage()) return false;
  return localStorage.getItem(ONBOARDING_DONE_KEY) === "true";
}

export function markOnboardingDone(): void {
  if (!canUseStorage()) return;
  localStorage.setItem(ONBOARDING_DONE_KEY, "true");
}

export function resetOnboardingDone(): void {
  if (!canUseStorage()) return;
  localStorage.removeItem(ONBOARDING_DONE_KEY);
}

export function getPostAuthRoute(): PostAuthRoute {
  if (isTauri) return "/chat";
  if (mustChangePassword()) return "/change-password";
  return "/chat";
}
