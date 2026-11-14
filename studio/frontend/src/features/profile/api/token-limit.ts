// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type TokenLimitPeriod = "day" | "week" | "month";

export type TokenLimitSettings = {
  enabled: boolean;
  period: TokenLimitPeriod;
  limit: number;
  timezone: string;
};

export type TokenLimitStatus = {
  settings: TokenLimitSettings;
  periodStart: number;
  resetsAt: number;
  used: number;
  reached: boolean;
  warning: boolean;
  override: "granted" | "used" | null;
};

export const TOKEN_LIMIT_CHANGED_EVENT = "unsloth:token-limit-changed";

async function requestTokenLimit(
  path: string,
  fallbackMessage: string,
  init?: RequestInit,
): Promise<TokenLimitStatus> {
  const res = await authFetch(`/api/token-limit${path}`, init);
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  const status = (await res.json()) as TokenLimitStatus;
  if (init) {
    window.dispatchEvent(
      new CustomEvent(TOKEN_LIMIT_CHANGED_EVENT, { detail: status }),
    );
  }
  return status;
}

export function fetchTokenLimitStatus(
  fallbackMessage: string,
): Promise<TokenLimitStatus> {
  return requestTokenLimit("/status", fallbackMessage);
}

export function updateTokenLimit(
  patch: Partial<TokenLimitSettings>,
  fallbackMessage: string,
): Promise<TokenLimitStatus> {
  return requestTokenLimit("", fallbackMessage, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

export function overrideTokenLimit(
  fallbackMessage: string,
): Promise<TokenLimitStatus> {
  return requestTokenLimit("/override", fallbackMessage, { method: "POST" });
}
