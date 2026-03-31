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
  tauriAutoAuth,
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

async function fetchAuthStatus(): Promise<AuthStatus> {
  try {
    const res = await fetch(apiUrl("/api/auth/status"));
    if (!res.ok) return { initialized: true, requires_password_change: mustChangePassword() };
    return (await res.json()) as AuthStatus;
  } catch {
    return { initialized: true, requires_password_change: mustChangePassword() };
  }
}

export async function requireAuth(): Promise<void> {
  // Tauri desktop: silently authenticate, skip login/change-password entirely
  if (isTauri && (await tauriAutoAuth())) return;

  if (await hasActiveSession()) {
    const { requires_password_change } = await fetchAuthStatus();
    if (requires_password_change || mustChangePassword()) {
      throw redirect({ to: "/change-password" });
    }
    return;
  }
  const status = await fetchAuthStatus();
  if (status.requires_password_change || mustChangePassword()) {
    throw redirect({ to: "/change-password" });
  }
  throw redirect({ to: status.initialized ? "/login" : "/change-password" });
}

export async function requireGuest(): Promise<void> {
  // Tauri: user should never land on /login, redirect to app
  if (isTauri && (await tauriAutoAuth())) {
    throw redirect({ to: getPostAuthRoute() });
  }
  if (!(await hasActiveSession())) return;
  throw redirect({ to: getPostAuthRoute() });
}

export async function requirePasswordChangeFlow(): Promise<void> {
  // Tauri: auto-auth handles password change silently
  if (isTauri && (await tauriAutoAuth())) {
    throw redirect({ to: getPostAuthRoute() });
  }

  const status = await fetchAuthStatus();
  if (status.requires_password_change || mustChangePassword()) return;
  if (await hasActiveSession()) {
    throw redirect({ to: getPostAuthRoute() });
  }
  throw redirect({ to: status.initialized ? "/login" : "/change-password" });
}
