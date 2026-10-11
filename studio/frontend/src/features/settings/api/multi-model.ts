// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export async function loadMultiModelEnabled(
  fallbackMessage: string,
): Promise<boolean> {
  const res = await authFetch("/api/settings/multi-model");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return Boolean((await res.json()).enabled);
}

export async function updateMultiModelEnabled(
  enabled: boolean,
  fallbackMessage: string,
): Promise<boolean> {
  const res = await authFetch("/api/settings/multi-model", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return Boolean((await res.json()).enabled);
}
