import { create } from "zustand";

import type { PlatformUser } from "./auth-types";

export const PLATFORM_AUTH_TOKEN_KEY = "rag-platform.auth-token";
export const PLATFORM_AUTH_SESSION_CLEARED_EVENT =
  "rag-platform:auth-session-cleared";

export type PlatformSessionStatus =
  | "anonymous"
  | "hydrating"
  | "authenticated"
  | "error";

interface PlatformSessionState {
  error: string | null;
  status: PlatformSessionStatus;
  user: PlatformUser | null;
  setError: (error: string) => void;
  setHydrating: () => void;
  setUser: (user: PlatformUser) => void;
  reset: () => void;
}

export const usePlatformSessionStore = create<PlatformSessionState>((set) => ({
  error: null,
  status: "anonymous",
  user: null,
  setError: (error) => set({ error, status: "error" }),
  setHydrating: () => set({ error: null, status: "hydrating" }),
  setUser: (user) => set({ error: null, status: "authenticated", user }),
  reset: () => set({ error: null, status: "anonymous", user: null }),
}));

function canUseBrowserStorage(): boolean {
  return typeof window !== "undefined" && typeof localStorage !== "undefined";
}

export function getPlatformSessionToken(): string | null {
  if (!canUseBrowserStorage()) return null;
  return localStorage.getItem(PLATFORM_AUTH_TOKEN_KEY);
}

export function hasPlatformSessionToken(): boolean {
  return Boolean(getPlatformSessionToken());
}

export function storePlatformSessionToken(token: string): void {
  const normalized = token.trim();
  if (!normalized || !canUseBrowserStorage()) return;
  localStorage.setItem(PLATFORM_AUTH_TOKEN_KEY, normalized);
  redirectingAfterUnauthorized = false;
}

export function clearPlatformSession(): void {
  if (canUseBrowserStorage()) {
    localStorage.removeItem(PLATFORM_AUTH_TOKEN_KEY);
    window.dispatchEvent(new Event(PLATFORM_AUTH_SESSION_CLEARED_EVENT));
  }
  usePlatformSessionStore.getState().reset();
}

let redirectingAfterUnauthorized = false;

export function clearPlatformSessionAndRedirectToLogin(): void {
  clearPlatformSession();
  if (redirectingAfterUnauthorized || typeof window === "undefined") return;
  if (window.location.pathname === "/login") return;
  redirectingAfterUnauthorized = true;
  window.history.replaceState(null, "", "/login");
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function resetPlatformUnauthorizedRedirectForTests(): void {
  redirectingAfterUnauthorized = false;
}
