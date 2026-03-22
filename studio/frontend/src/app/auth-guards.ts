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

async function checkAuthInitialized(): Promise<boolean> {
  try {
    const res = await fetch(apiUrl("/api/auth/status"));
    if (!res.ok) return true; // fallback to login on error
    const data = (await res.json()) as { initialized: boolean };
    return data.initialized;
  } catch {
    return true; // fallback to login on error
  }
}

async function checkPasswordChangeRequired(): Promise<boolean> {
  try {
    const res = await fetch(apiUrl("/api/auth/status"));
    if (!res.ok) return mustChangePassword();
    const data = (await res.json()) as { requires_password_change: boolean };
    return data.requires_password_change || mustChangePassword();
  } catch {
    return mustChangePassword();
  }
}

export async function requireAuth(): Promise<void> {
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
  if (!(await hasActiveSession())) return;
  throw redirect({ to: getPostAuthRoute() });
}

export async function requirePasswordChangeFlow(): Promise<void> {
  const requiresPasswordChange = await checkPasswordChangeRequired();

  if (requiresPasswordChange) return;

  if (await hasActiveSession()) {
    throw redirect({ to: getPostAuthRoute() });
  }

  const initialized = await checkAuthInitialized();
  throw redirect({ to: initialized ? "/login" : "/change-password" });
}
