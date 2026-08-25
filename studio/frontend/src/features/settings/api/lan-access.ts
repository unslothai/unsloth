// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  type LanAccessStatus,
  normalizeLanAccessStatus,
} from "./lan-access-state";

export type { LanAccessStatus } from "./lan-access-state";

async function requestLanAccess(
  path = "",
  init?: RequestInit,
): Promise<LanAccessStatus> {
  const response = await authFetch(`/api/settings/lan-access${path}`, init);
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "LAN access request failed"),
    );
  }
  return normalizeLanAccessStatus(await response.json());
}

export const loadLanAccess = () => requestLanAccess();
export const startLanAccess = () =>
  requestLanAccess("/start", { method: "POST" });
export const stopLanAccess = () =>
  requestLanAccess("/stop", { method: "POST" });
export const updateLanAccessAutoStart = (enabled: boolean) =>
  requestLanAccess("/auto-start", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
