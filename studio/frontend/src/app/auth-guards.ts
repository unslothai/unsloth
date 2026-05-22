// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { redirect } from "@tanstack/react-router";
import { apiUrl, isTauri } from "@/lib/api-base";
import {
  getAuthToken,
  getPostAuthRoute,
  hasAuthToken,
  hasRefreshToken,
  mustChangePassword,
  refreshSession,
  setMustChangePassword,
} from "@/features/auth";

async function hasActiveSession(): Promise<boolean> {
  if (hasAuthToken()) return true;
  if (!hasRefreshToken()) return false;
  return refreshSession();
}

interface AuthStatus {
  initialized: boolean;
  requires_password_change: boolean;
}

// beforeLoad runs this on every navigation. A short TTL keeps rapid in-app
// navigation from waiting on a round-trip each time. The key folds in the auth
// token and the must-change flag, so login/logout/refresh/password-change all
// invalidate it automatically; only successful responses are cached, so a
// stopped backend is retried on the next navigation instead of masked.
const AUTH_STATUS_TTL_MS = 30_000;
let authStatusCache: { key: string; status: AuthStatus; expiresAt: number } | null =
  null;

function authStatusKey(): string {
  return `${getAuthToken() ?? ""}|${mustChangePassword() ? 1 : 0}`;
}

async function fetchAuthStatus(): Promise<AuthStatus> {
  const cached = authStatusCache;
  if (cached && cached.key === authStatusKey() && Date.now() < cached.expiresAt) {
    return cached.status;
  }
  try {
    const res = await fetch(apiUrl("/api/auth/status"));
    if (!res.ok) return { initialized: true, requires_password_change: mustChangePassword() };
    const status = (await res.json()) as AuthStatus;
    // Server truth wins; keep localStorage in sync both ways.
    if (status.requires_password_change !== mustChangePassword()) {
      setMustChangePassword(status.requires_password_change);
    }
    authStatusCache = {
      key: authStatusKey(),
      status,
      expiresAt: Date.now() + AUTH_STATUS_TTL_MS,
    };
    return status;
  } catch {
    return { initialized: true, requires_password_change: mustChangePassword() };
  }
}

function authRedirect(to: "/login" | "/change-password"): never {
  throw redirect({ to });
}

export async function requireAuth(): Promise<void> {
  if (isTauri) {
    // AppProvider owns backend startup + desktop auth; route guards run before it mounts.
    return;
  }

  if (await hasActiveSession()) {
    const { requires_password_change } = await fetchAuthStatus();
    if (requires_password_change || mustChangePassword()) {
      authRedirect("/change-password");
    }
    return;
  }
  const status = await fetchAuthStatus();
  if (status.requires_password_change || mustChangePassword()) {
    authRedirect("/change-password");
  }
  authRedirect(status.initialized ? "/login" : "/change-password");
}

export async function requireGuest(): Promise<void> {
  if (isTauri) {
    throw redirect({ to: "/chat" });
  }
  if (!(await hasActiveSession())) return;
  // Reconcile localStorage before routing.
  await fetchAuthStatus();
  throw redirect({ to: getPostAuthRoute() });
}

export async function requirePasswordChangeFlow(): Promise<void> {
  if (isTauri) {
    throw redirect({ to: "/chat" });
  }

  const status = await fetchAuthStatus();
  if (status.requires_password_change || mustChangePassword()) return;
  if (await hasActiveSession()) {
    throw redirect({ to: getPostAuthRoute() });
  }
  authRedirect(status.initialized ? "/login" : "/change-password");
}
