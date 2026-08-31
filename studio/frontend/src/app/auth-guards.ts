// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { redirect } from "@tanstack/react-router";
import { apiUrl, isTauri } from "@/lib/api-base";
import {
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

const AUTH_STATUS_TTL_MS = 30_000;
let authStatusCheckedAt = 0;
let authStatusRequest: Promise<AuthStatus> | null = null;

function hasFreshAuthStatus(): boolean {
  return (
    authStatusCheckedAt !== 0 &&
    Date.now() - authStatusCheckedAt < AUTH_STATUS_TTL_MS
  );
}

async function fetchAuthStatus(): Promise<AuthStatus> {
  if (authStatusRequest) return authStatusRequest;

  const request = (async () => {
    try {
      const res = await fetch(apiUrl("/api/auth/status"));
      if (!res.ok) {
        return {
          initialized: true,
          requires_password_change: mustChangePassword(),
        };
      }
      const status = (await res.json()) as AuthStatus;
      authStatusCheckedAt = Date.now();
      // Server truth wins; keep localStorage in sync both ways.
      if (status.requires_password_change !== mustChangePassword()) {
        setMustChangePassword(status.requires_password_change);
      }
      return status;
    } catch {
      return {
        initialized: true,
        requires_password_change: mustChangePassword(),
      };
    }
  })().finally(() => {
    authStatusRequest = null;
  });
  authStatusRequest = request;
  return request;
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
    // Reconcile periodically so local-only routes cannot outlive a server-side
    // password-change requirement, while nearby route switches stay local.
    if (mustChangePassword() || !hasFreshAuthStatus()) {
      const { requires_password_change } = await fetchAuthStatus();
      if (requires_password_change || mustChangePassword()) {
        authRedirect("/change-password");
      }
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
