// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
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
  account_id: string;
  username: string;
  setup_code: string;
  expires_at: string;
}
// The backend nests the account and names the expiry after the code it belongs to.
interface AccountSetupResponse {
  account: StudioAccount;
  setup_code: string;
  setup_code_expires_at: string;
}
function toSetupCode(payload: AccountSetupResponse): AccountSetupCode {
  return {
    account_id: payload.account.account_id,
    username: payload.account.username,
    setup_code: payload.setup_code,
    expires_at: payload.setup_code_expires_at,
  };
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
const accountPath = (accountId: string) => `/${encodeURIComponent(accountId)}`;
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
  const result = toSetupCode((await response.json()) as AccountSetupResponse);
  await refreshAccountPolicy();
  return result;
}
export async function regenerateSetupCode(
  accountId: string,
): Promise<AccountSetupCode> {
  const response = await accountsRequest(
    `${accountPath(accountId)}/setup-code`,
    { method: "POST" },
  );
  return toSetupCode((await response.json()) as AccountSetupResponse);
}
export async function setAccountActive(
  accountId: string,
  active: boolean,
): Promise<void> {
  await accountsRequest(accountPath(accountId), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_active: active }),
  });
  await refreshAccountPolicy();
}
export async function deleteAccount(accountId: string): Promise<void> {
  await accountsRequest(accountPath(accountId), { method: "DELETE" });
  await refreshAccountPolicy();
}
