// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { redirect } from "@tanstack/react-router";
import {
  getPostAuthRoute,
  hasAuthToken,
  hasRefreshToken,
  mustChangePassword,
  refreshSession,
} from "@/features/auth";

type AuthStatus = {
  initialized: boolean;
  requires_password_change: boolean;
  auth_disabled?: boolean;
};

async function getAuthStatus(): Promise<AuthStatus | null> {
  try {
    const res = await fetch("/api/auth/status");
    if (!res.ok) return null;
    return (await res.json()) as AuthStatus;
  } catch {
    return null;
  }
}

async function hasActiveSession(): Promise<boolean> {
  if (hasAuthToken()) return true;
  if (!hasRefreshToken()) return false;
  return refreshSession();
}

async function checkAuthInitialized(): Promise<boolean> {
  const status = await getAuthStatus();
  if (status?.auth_disabled) return true;
  return status?.initialized ?? true; // fallback to login on error
}

async function checkPasswordChangeRequired(): Promise<boolean> {
  const status = await getAuthStatus();
  if (status?.auth_disabled) return false;
  if (!status) return mustChangePassword();
  return status.requires_password_change || mustChangePassword();
}

export async function requireAuth(): Promise<void> {
  const status = await getAuthStatus();
  if (status?.auth_disabled) return;

  if (await hasActiveSession()) {
    if (await checkPasswordChangeRequired()) {
      throw redirect({ to: "/change-password" });
    }
    return;
  }
  const requiresPasswordChange = await checkPasswordChangeRequired();
  if (requiresPasswordChange) throw redirect({ to: "/change-password" });
  const initialized = await checkAuthInitialized();
  throw redirect({ to: initialized ? "/login" : "/change-password" });
}

export async function requireGuest(): Promise<void> {
  const status = await getAuthStatus();
  if (status?.auth_disabled) {
    throw redirect({ to: getPostAuthRoute() });
  }

  if (!(await hasActiveSession())) return;
  throw redirect({ to: getPostAuthRoute() });
}

export async function requirePasswordChangeFlow(): Promise<void> {
  const status = await getAuthStatus();
  if (status?.auth_disabled) {
    throw redirect({ to: getPostAuthRoute() });
  }

  const requiresPasswordChange = await checkPasswordChangeRequired();

  if (requiresPasswordChange) return;

  if (await hasActiveSession()) {
    throw redirect({ to: getPostAuthRoute() });
  }

  const initialized = await checkAuthInitialized();
  throw redirect({ to: initialized ? "/login" : "/change-password" });
}
