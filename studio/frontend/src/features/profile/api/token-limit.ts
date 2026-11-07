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
  override: "granted" | "used" | null;
};

export type TokenLimitStatus = {
  settings: TokenLimitSettings;
  periodStart: number;
  resetsAt: number;
  pools: TokenLimitPoolStatus[];
};

export type TokenLimitUpdate = Partial<
  Omit<TokenLimitSettings, "passwordProtected">
> & {
  password?: string;
  /** Sets or changes the password; an empty string removes it. */
  newPassword?: string;
};

export const TOKEN_LIMIT_CHANGED_EVENT = "unsloth:token-limit-changed";

async function requestTokenLimit(
  path: string,
  fallbackMessage: string,
  method: "GET" | "PUT" | "POST" = "GET",
  body?: object,
): Promise<TokenLimitStatus> {
  const res = await authFetch(
    `/api/token-limit${path}`,
    method === "GET"
      ? undefined
      : {
          method,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  const status = (await res.json()) as TokenLimitStatus;
  if (method !== "GET") {
    window.dispatchEvent(
      new CustomEvent(TOKEN_LIMIT_CHANGED_EVENT, { detail: status }),
    );
  }
  return status;
}

export async function fetchTokenLimitStatus(
  fallbackMessage: string,
): Promise<TokenLimitStatus> {
  return requestTokenLimit("/status", fallbackMessage);
}

export async function updateTokenLimit(
  patch: TokenLimitUpdate,
  fallbackMessage: string,
): Promise<TokenLimitStatus> {
  return requestTokenLimit("", fallbackMessage, "PUT", patch);
}

export async function overrideTokenLimit(
  pool: TokenLimitPool,
  fallbackMessage: string,
  password?: string,
): Promise<TokenLimitStatus> {
  return requestTokenLimit("/override", fallbackMessage, "POST", {
    pool,
    password,
  });
}
