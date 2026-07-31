// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
// This header helper is API-layer-only and is not part of the feature's
// React-facing public barrel.
// eslint-disable-next-line no-restricted-imports
import { hubTokenHeader } from "@/features/hub/lib/hub-token-header";

export type HfTokenValidationStatus =
  | "missing"
  | "valid"
  | "invalid"
  | "rate_limited"
  | "unavailable";

export interface HfTokenValidationResult {
  status: HfTokenValidationStatus;
  retryAfterSeconds: number | null;
}

export async function validateHfToken(
  token: string | null | undefined,
  signal?: AbortSignal,
): Promise<HfTokenValidationResult> {
  const normalized = token?.trim() ?? "";
  if (!normalized) {
    return { status: "missing", retryAfterSeconds: null };
  }
  const response = await authFetch("/api/hub/token/validate", {
    method: "POST",
    signal,
    headers: hubTokenHeader(normalized),
  });
  if (!response.ok) {
    return { status: "unavailable", retryAfterSeconds: null };
  }
  const body = (await response.json()) as {
    status?: HfTokenValidationStatus;
    retry_after_seconds?: number | null;
  };
  return {
    status: body.status ?? "unavailable",
    retryAfterSeconds: body.retry_after_seconds ?? null,
  };
}
