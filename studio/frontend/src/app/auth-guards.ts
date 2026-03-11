// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { redirect } from "@tanstack/react-router";
import {
  getPostAuthRoute,
  hasAuthToken,
  hasRefreshToken,
  refreshSession,
} from "@/features/auth";

async function hasActiveSession(): Promise<boolean> {
  if (hasAuthToken()) return true;
  if (!hasRefreshToken()) return false;
  return refreshSession();
}

async function checkAuthInitialized(): Promise<boolean> {
  try {
    const res = await fetch("/api/auth/status");
    if (!res.ok) return true; // fallback to login on error
    const data = (await res.json()) as { initialized: boolean };
    return data.initialized;
  } catch {
    return true; // fallback to login on error
  }
}

export async function requireAuth(): Promise<void> {
  if (await hasActiveSession()) return;
  const initialized = await checkAuthInitialized();
  throw redirect({ to: initialized ? "/login" : "/signup" });
}

export async function requireGuest(): Promise<void> {
  if (!(await hasActiveSession())) return;
  throw redirect({ to: getPostAuthRoute() });
}
