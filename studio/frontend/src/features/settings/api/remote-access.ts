// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  type RemoteAccessStatus,
  type TunnelMode,
  normalizeRemoteAccessStatus,
} from "./remote-access-state";

export type { RemoteAccessStatus } from "./remote-access-state";

async function requestRemoteAccess(
  path = "",
  init?: RequestInit,
): Promise<RemoteAccessStatus> {
  const response = await authFetch(`/api/settings/remote-access${path}`, init);
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Remote access request failed"),
    );
  }
  return normalizeRemoteAccessStatus(await response.json());
}

export const loadRemoteAccess = () => requestRemoteAccess();
export const startRemoteAccess = () =>
  requestRemoteAccess("/start", { method: "POST" });
export const stopRemoteAccess = () =>
  requestRemoteAccess("/stop", { method: "POST" });
export const updateRemoteAccessAutoStart = (enabled: boolean) =>
  requestRemoteAccess("/auto-start", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });

export const updateRemoteAccessConfig = (config: {
  mode: TunnelMode;
  hostname?: string;
  token?: string;
}) =>
  requestRemoteAccess("/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });

export const setupNamedTunnel = (hostname: string, reauthenticate = false) =>
  requestRemoteAccess("/named/setup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hostname, reauthenticate }),
  });

export const confirmNamedTunnelDns = (
  operationId: string,
  hostname: string,
  tunnelId: string,
) =>
  requestRemoteAccess("/named/setup/confirm-dns", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: API schema
    body: JSON.stringify({
      operation_id: operationId,
      hostname,
      tunnel_id: tunnelId,
    }),
  });

export const cancelNamedTunnelSetup = (operationId: string) =>
  requestRemoteAccess("/named/setup/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: API schema
    body: JSON.stringify({ operation_id: operationId }),
  });

export const removeNamedTunnel = (configurationId: string) =>
  requestRemoteAccess("/named", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: API schema
    body: JSON.stringify({ configuration_id: configurationId }),
  });
