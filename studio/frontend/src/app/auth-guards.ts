// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { redirect } from "@tanstack/react-router";
import { apiUrl } from "@/lib/api-base";
import {
  getPostAuthRoute,
  hasAuthToken,
  hasRefreshToken,
  mustChangePassword,
  refreshSession,
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
  if (!(await hasActiveSession())) return;
  throw redirect({ to: getPostAuthRoute() });
}

export async function requirePasswordChangeFlow(): Promise<void> {
  const status = await fetchAuthStatus();

  if (status.requires_password_change || mustChangePassword()) return;

  if (await hasActiveSession()) {
    throw redirect({ to: getPostAuthRoute() });
  }

  throw redirect({ to: status.initialized ? "/login" : "/change-password" });
}
