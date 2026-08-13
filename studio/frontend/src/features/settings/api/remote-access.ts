// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  type RemoteAccessStatus,
  normalizeRemoteAccessStatus,
} from "./remote-access-state";

export type { RemoteAccessStatus } from "./remote-access-state";

export type RemoteAccessRequest = {
  path: string;
  init?: RequestInit;
};

const post = (path: string): RemoteAccessRequest => ({
  path,
  init: { method: "POST" },
});

export const remoteAccessStartRequest = (): RemoteAccessRequest =>
  post("/start");

export const remoteAccessStopRequest = (): RemoteAccessRequest => post("/stop");

export const remoteAccessAutoStartRequest = (
  enabled: boolean,
): RemoteAccessRequest => ({
  path: "/auto-start",
  init: {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  },
});

export const remoteAccessMethodRequest = (
  method: RemoteAccessStatus["method"],
): RemoteAccessRequest => ({
  path: "/method",
  init: {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method }),
  },
});

export const remoteAccessProvisionRequest = (
  hostname: string,
): RemoteAccessRequest => ({
  path: "/custom/provision",
  init: {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hostname }),
  },
});

export const remoteAccessCancelRequest = (
  expectedRevision: number,
): RemoteAccessRequest => ({
  path: "/custom/cancel",
  init: {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: API schema
    body: JSON.stringify({ expected_revision: expectedRevision }),
  },
});

export const remoteAccessTeardownRequest = (): RemoteAccessRequest =>
  post("/custom/teardown");

async function requestRemoteAccess(
  request: RemoteAccessRequest = { path: "" },
): Promise<RemoteAccessStatus> {
  const response = await authFetch(
    `/api/settings/remote-access${request.path}`,
    request.init,
  );
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Remote access request failed"),
    );
  }
  return normalizeRemoteAccessStatus(await response.json());
}

export const loadRemoteAccess = () => requestRemoteAccess();
export const startRemoteAccess = () =>
  requestRemoteAccess(remoteAccessStartRequest());
export const stopRemoteAccess = () =>
  requestRemoteAccess(remoteAccessStopRequest());
export const updateRemoteAccessAutoStart = (enabled: boolean) =>
  requestRemoteAccess(remoteAccessAutoStartRequest(enabled));
export const updateRemoteAccessMethod = (
  method: RemoteAccessStatus["method"],
) => requestRemoteAccess(remoteAccessMethodRequest(method));
export const provisionCustomRemoteAccess = (hostname: string) =>
  requestRemoteAccess(remoteAccessProvisionRequest(hostname));
export const cancelCustomRemoteAccess = (expectedRevision: number) =>
  requestRemoteAccess(remoteAccessCancelRequest(expectedRevision));
export const teardownCustomRemoteAccess = () =>
  requestRemoteAccess(remoteAccessTeardownRequest());
