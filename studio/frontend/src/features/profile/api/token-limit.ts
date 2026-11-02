// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type TokenLimitPeriod = "day" | "week" | "month";
export type TokenLimitMode = "combined" | "split";
export type TokenLimitMethod = "total" | "incremental";
export type TokenLimitPool = "combined" | "local" | "api";

export type TokenLimitSettings = {
  enabled: boolean;
  period: TokenLimitPeriod;
  mode: TokenLimitMode;
  method: TokenLimitMethod;
  limit: number;
  localLimit: number;
  apiLimit: number;
  timezone: string;
  passwordProtected: boolean;
};

export type TokenLimitPoolStatus = {
  pool: TokenLimitPool;
  used: number;
  limit: number;
  reached: boolean;
  warning: boolean;
  /** This period's one-time override: granted and waiting, used, or not taken. */
  override: "granted" | "used" | null;
};

export type TokenLimitStatus = {
  settings: TokenLimitSettings;
  periodStart: number;
  resetsAt: number;
  /** Empty while the limiter is off. */
  pools: TokenLimitPoolStatus[];
};

export type TokenLimitUpdate = Partial<
  Omit<TokenLimitSettings, "passwordProtected">
> & {
  password?: string;
  /** Sets or changes the password; an empty string removes it. */
  newPassword?: string;
};

/** Fired after a save or an override so every open view refetches at once. */
export const TOKEN_LIMIT_CHANGED_EVENT = "unsloth:token-limit-changed";

function announce(status: TokenLimitStatus): TokenLimitStatus {
  window.dispatchEvent(
    new CustomEvent(TOKEN_LIMIT_CHANGED_EVENT, { detail: status }),
  );
  return status;
}

export async function fetchTokenLimitStatus(
  fallbackMessage: string,
): Promise<TokenLimitStatus> {
  const res = await authFetch("/api/token-limit/status");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return (await res.json()) as TokenLimitStatus;
}

export async function updateTokenLimit(
  patch: TokenLimitUpdate,
  fallbackMessage: string,
): Promise<TokenLimitStatus> {
  const res = await authFetch("/api/token-limit", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return announce((await res.json()) as TokenLimitStatus);
}

export async function overrideTokenLimit(
  pool: TokenLimitPool,
  fallbackMessage: string,
): Promise<TokenLimitStatus> {
  const res = await authFetch("/api/token-limit/override", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pool }),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return announce((await res.json()) as TokenLimitStatus);
}
