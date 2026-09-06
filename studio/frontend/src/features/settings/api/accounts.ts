// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";
import { fetchAuthStatus } from "@/features/auth/login-client";
import { normalizeAccountUsername } from "@/lib/account-transition";

export interface StudioAccount {
  account_id: string;
  username: string;
  role: "owner" | "user";
  is_active: boolean;
  created_at: string;
}
export interface AccountSetupCode {
  username: string;
  setup_code: string;
  expires_at: string;
}

async function accountsRequest(
  path = "",
  init?: RequestInit,
): Promise<Response> {
  const response = await authFetch(`/api/accounts${path}`, init);
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      detail?: unknown;
    } | null;
    throw new Error(
      typeof payload?.detail === "string"
        ? payload.detail
        : "Account request failed.",
    );
  }
  return response;
}
const accountPath = (username: string) =>
  `/${encodeURIComponent(normalizeAccountUsername(username))}`;
// Recheck installation policy after mutations; a failed status read must not lose a one-time code.
async function refreshAccountPolicy(): Promise<void> {
  await fetchAuthStatus().catch(() => undefined);
}
export async function fetchAccounts(): Promise<StudioAccount[]> {
  const response = await accountsRequest();
  const data = (await response.json()) as { accounts: StudioAccount[] };
  return data.accounts;
}
export async function createAccount(
  username: string,
): Promise<AccountSetupCode> {
  const response = await accountsRequest("", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: normalizeAccountUsername(username) }),
  });
  const result = (await response.json()) as AccountSetupCode;
  await refreshAccountPolicy();
  return result;
}
export async function regenerateSetupCode(
  username: string,
): Promise<AccountSetupCode> {
  const response = await accountsRequest(
    `${accountPath(username)}/setup-code`,
    { method: "POST" },
  );
  return response.json();
}
export async function setAccountActive(
  username: string,
  active: boolean,
): Promise<void> {
  await accountsRequest(
    `${accountPath(username)}/${active ? "reactivate" : "deactivate"}`,
    { method: "POST" },
  );
  await refreshAccountPolicy();
}
export async function deleteAccount(username: string): Promise<void> {
  await accountsRequest(accountPath(username), { method: "DELETE" });
  await refreshAccountPolicy();
}
