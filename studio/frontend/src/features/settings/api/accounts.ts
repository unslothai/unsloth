// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export interface CurrentAccount {
  username: string;
  is_admin: boolean;
}

export interface ManagedAccount extends CurrentAccount {
  must_change_password: boolean;
  setup_code_expires_at: string | null;
  setup_code_expired: boolean;
}

export interface CreatedManagedAccount extends ManagedAccount {
  setup_code: string;
}

async function detail(response: Response, fallback: string): Promise<string> {
  const body = (await response.json().catch(() => null)) as {
    detail?: string;
  } | null;
  return body?.detail ?? fallback;
}

export async function fetchCurrentAccount(): Promise<CurrentAccount> {
  const response = await authFetch("/api/auth/me");
  if (!response.ok)
    throw new Error(await detail(response, "Failed to load account"));
  return response.json();
}

export async function fetchAccounts(): Promise<ManagedAccount[]> {
  const response = await authFetch("/api/auth/users");
  if (!response.ok)
    throw new Error(await detail(response, "Failed to load accounts"));
  const body = (await response.json()) as { users: ManagedAccount[] };
  return body.users;
}

export async function createAccount(
  username: string,
): Promise<CreatedManagedAccount> {
  const response = await authFetch("/api/auth/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username }),
  });
  if (!response.ok)
    throw new Error(await detail(response, "Failed to create account"));
  return response.json();
}

export async function regenerateSetupCode(
  username: string,
): Promise<CreatedManagedAccount> {
  const response = await authFetch(
    `/api/auth/users/${encodeURIComponent(username)}/setup-code`,
    { method: "POST" },
  );
  if (!response.ok)
    throw new Error(await detail(response, "Failed to regenerate setup code"));
  return response.json();
}

export async function deleteAccount(username: string): Promise<void> {
  const response = await authFetch(
    `/api/auth/users/${encodeURIComponent(username)}`,
    {
      method: "DELETE",
    },
  );
  if (!response.ok)
    throw new Error(await detail(response, "Failed to delete account"));
}
