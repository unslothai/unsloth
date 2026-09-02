// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { redirect } from "@tanstack/react-router";
import { apiUrl, isTauri } from "@/lib/api-base";
import {
  authFetch,
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
let authStatusRequest: Promise<AuthStatus> | null = null;

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
      // /status describes the seeded installation owner, while the local flag
      // can describe any authenticated managed account. A false owner status
      // must not clear another account's forced-change route, and a true one
      // must not impose the owner's recovery on every signed-in managed session:
      // that traps them on /change-password until the owner finishes, and the
      // next refresh re-sets the flag even after they change their own password.
      if (
        status.requires_password_change &&
        !mustChangePassword() &&
        !hasAuthToken()
      ) {
        setMustChangePassword(true);
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

let accountStatusCheckedAt = 0;
let accountStatusRequest: Promise<boolean> | null = null;

function hasFreshAccountStatus(): boolean {
  return (
    accountStatusCheckedAt !== 0 &&
    Date.now() - accountStatusCheckedAt < AUTH_STATUS_TTL_MS
  );
}

/** Whether THIS signed-in account still owes a password change. */
async function fetchAccountMustChangePassword(): Promise<boolean> {
  if (accountStatusRequest) return accountStatusRequest;

  const request = (async () => {
    try {
      const res = await authFetch("/api/auth/me");
      if (!res.ok) return mustChangePassword();
      const me = (await res.json()) as { must_change_password?: boolean };
      accountStatusCheckedAt = Date.now();
      const required = me.must_change_password === true;
      if (required !== mustChangePassword()) setMustChangePassword(required);
      return required;
    } catch {
      return mustChangePassword();
    }
  })().finally(() => {
    accountStatusRequest = null;
  });
  accountStatusRequest = request;
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
    // Against /me, not /status: /status is unauthenticated and describes the
    // installation owner, so while the owner is in recovery it would send every
    // other signed-in account to /change-password, and keep sending them there
    // after they had changed their own password, until the owner finished.
    if (mustChangePassword() || !hasFreshAccountStatus()) {
      if (await fetchAccountMustChangePassword()) {
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
  // Reconcile localStorage before routing, from this account's own state.
  await fetchAccountMustChangePassword();
  throw redirect({ to: getPostAuthRoute() });
}

export async function requirePasswordChangeFlow(): Promise<void> {
  if (isTauri) {
    throw redirect({ to: "/chat" });
  }

  if (await hasActiveSession()) {
    // Signed in: only this account's own requirement keeps it on this page. The
    // owner being in recovery is not a reason to hold anybody else here.
    if (await fetchAccountMustChangePassword()) return;
    throw redirect({ to: getPostAuthRoute() });
  }

  const status = await fetchAuthStatus();
  if (status.requires_password_change || mustChangePassword()) return;
  authRedirect(status.initialized ? "/login" : "/change-password");
}
