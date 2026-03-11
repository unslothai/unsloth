// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

export const AUTH_TOKEN_KEY = "unsloth_auth_token";
export const AUTH_REFRESH_TOKEN_KEY = "unsloth_auth_refresh_token";
export const ONBOARDING_DONE_KEY = "unsloth_onboarding_done";

type PostAuthRoute = "/onboarding" | "/studio";

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
  if (!canUseStorage()) return;
  localStorage.setItem(AUTH_TOKEN_KEY, accessToken);
  localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, refreshToken);
}

export function clearAuthTokens(): void {
  if (!canUseStorage()) return;
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_REFRESH_TOKEN_KEY);
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
  return isOnboardingDone() ? "/studio" : "/onboarding";
}
