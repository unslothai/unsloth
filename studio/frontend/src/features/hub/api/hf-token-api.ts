// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";

export interface SavedHfTokenResponse {
  token: string | null;
  has_token: boolean;
}

async function parse(response: Response): Promise<SavedHfTokenResponse> {
  const body = (await response.json().catch(() => null)) as
    | SavedHfTokenResponse
    | { detail?: unknown }
    | null;
  if (!response.ok) {
    const detail =
      body && "detail" in body ? formatFastApiDetail(body.detail) : null;
    throw new Error(
      detail || `Hugging Face credential request failed (${response.status})`,
    );
  }
  return body as SavedHfTokenResponse;
}

export async function loadSavedHfToken(): Promise<SavedHfTokenResponse> {
  return parse(await authFetch("/api/settings/hugging-face-token"));
}

export async function saveHfToken(
  token: string,
): Promise<SavedHfTokenResponse> {
  return parse(
    await authFetch("/api/settings/hugging-face-token", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    }),
  );
}

export async function clearSavedHfToken(): Promise<SavedHfTokenResponse> {
  return parse(
    await authFetch("/api/settings/hugging-face-token", { method: "DELETE" }),
  );
}
