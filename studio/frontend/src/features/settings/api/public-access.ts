// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  type PublicAccessStatus,
  normalizePublicAccessStatus,
} from "./public-access-state";

export type { PublicAccessStatus } from "./public-access-state";

async function requestPublicAccess(
  path = "",
  init?: RequestInit,
): Promise<PublicAccessStatus> {
  const response = await authFetch(`/api/settings/public-access${path}`, init);
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Public access request failed"),
    );
  }
  return normalizePublicAccessStatus(await response.json());
}

export const loadPublicAccess = () => requestPublicAccess();
export const startPublicAccess = () =>
  requestPublicAccess("/start", { method: "POST" });
export const stopPublicAccess = () =>
  requestPublicAccess("/stop", { method: "POST" });
export const updatePublicAccessAutoStart = (enabled: boolean) =>
  requestPublicAccess("/auto-start", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
