// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";

export interface ApiKey {
  id: number;
  name: string;
  key_prefix: string;
  created_at: string;
  last_used_at: string | null;
  expires_at: string | null;
  is_active: boolean;
}

/** a refused create or revoke, carrying the server's own reason when it gave one */
export class ApiKeyActionError extends Error {}

async function actionError(res: Response): Promise<ApiKeyActionError> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    return new ApiKeyActionError(
      typeof body.detail === "string" ? body.detail : "",
    );
  } catch {
    return new ApiKeyActionError("");
  }
}

export async function fetchApiKeys(): Promise<ApiKey[]> {
  const res = await authFetch("/api/auth/api-keys");
  if (!res.ok) throw new Error("Failed to load API access");
  const data = (await res.json()) as { api_keys: ApiKey[] };
  return data.api_keys.filter((k) => k.is_active);
}

export async function createApiKey(
  name: string,
  expiresInDays: number | null,
): Promise<{ key: string; api_key: ApiKey }> {
  const res = await authFetch("/api/auth/api-keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, expires_in_days: expiresInDays }),
  });
  if (!res.ok) throw await actionError(res);
  return res.json();
}

export async function revokeApiKey(keyId: number): Promise<void> {
  const res = await authFetch(`/api/auth/api-keys/${keyId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw await actionError(res);
}
