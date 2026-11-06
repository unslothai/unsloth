// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  type LanAccessStatus,
  lanApiUrls,
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

/** Publishes only on change: each poll's fresh array re-rendered every reader every 5 s. */
export function publishLanUrls(status: LanAccessStatus): void {
  const next = lanApiUrls(status);
  const current = usePlatformStore.getState().lanUrls;
  if (
    next.length === current.length &&
    next.every((url, i) => url === current[i])
  ) {
    return;
  }
  usePlatformStore.setState({ lanUrls: next });
}
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

export const updateLanAccessPort = (port: number | null) =>
  requestLanAccess("/port", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ port }),
  });

export const updateLanAccessAddresses = (addresses: string[] | null) =>
  requestLanAccess("/addresses", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ addresses }),
  });
