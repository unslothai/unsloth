// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { isTauri } from "@/lib/api-base";

export const AUTH_TOKEN_KEY = "unsloth_auth_token";
export const AUTH_REFRESH_TOKEN_KEY = "unsloth_auth_refresh_token";
export const ONBOARDING_DONE_KEY = "unsloth_onboarding_done";
export const AUTH_MUST_CHANGE_PASSWORD_KEY = "unsloth_auth_must_change_password";

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
  mustChangePassword = false,
): void {
  if (!canUseStorage()) return;
  localStorage.setItem(AUTH_TOKEN_KEY, accessToken);
  localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, refreshToken);
  localStorage.setItem(AUTH_MUST_CHANGE_PASSWORD_KEY, String(mustChangePassword));
}

export function clearAuthTokens(): void {
  if (!canUseStorage()) return;
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_REFRESH_TOKEN_KEY);
  localStorage.removeItem(AUTH_MUST_CHANGE_PASSWORD_KEY);
}

export function mustChangePassword(): boolean {
  if (!canUseStorage()) return false;
  return localStorage.getItem(AUTH_MUST_CHANGE_PASSWORD_KEY) === "true";
}

export function setMustChangePassword(required: boolean): void {
  if (!canUseStorage()) return;
  localStorage.setItem(AUTH_MUST_CHANGE_PASSWORD_KEY, String(required));
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
  if (usePlatformStore.getState().isChatOnly()) return "/chat";
  return "/chat";
}
