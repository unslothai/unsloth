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

export async function requireAuth(): Promise<void> {
  if (await hasActiveSession()) return;
  throw redirect({ to: "/login" });
}

export async function requireGuest(): Promise<void> {
  if (!(await hasActiveSession())) return;
  throw redirect({ to: getPostAuthRoute() });
}
