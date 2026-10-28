// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type FirstTokenTimeoutSettings = {
  minutes: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_minutes: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  max_minutes: number;
};

export async function firstTokenTimeoutSettings(
  fallbackMessage: string,
  minutes?: number,
): Promise<FirstTokenTimeoutSettings> {
  const response = await authFetch("/api/settings/first-token-timeout", {
    method: minutes === undefined ? "GET" : "PUT",
    ...(minutes === undefined
      ? {}
      : {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ minutes }),
        }),
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response, fallbackMessage));
  }
  return response.json();
}
