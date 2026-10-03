// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

async function adminRequest(path: string, init?: RequestInit): Promise<Response> {
  const response = await authFetch(`/api/admin${path}`, init);
  if (!response.ok) {
    throw new Error(await readFastApiError(response, "Admin request failed"));
  }
  return response;
}

export async function fetchBlockedModels(): Promise<string[]> {
  const response = await adminRequest("/model-policy");
  return ((await response.json()) as { blocked_models: string[] }).blocked_models;
}

export async function saveBlockedModels(blocked: string[]): Promise<string[]> {
  const response = await adminRequest("/model-policy", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blocked_models: blocked }),
  });
  return ((await response.json()) as { blocked_models: string[] }).blocked_models;
}
