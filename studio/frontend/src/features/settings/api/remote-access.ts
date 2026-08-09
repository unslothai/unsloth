// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  type RemoteAccessKind,
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

export const remoteAccessStartRequest = (
  kind?: RemoteAccessKind,
): RemoteAccessRequest => post(kind ? `/start?kind=${kind}` : "/start");

export const remoteAccessStopRequest = (): RemoteAccessRequest => post("/stop");

export const remoteAccessAutoStartRequest = (
  enabled: boolean,
  kind?: RemoteAccessKind,
): RemoteAccessRequest => ({
  path: "/auto-start",
  init: {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(kind ? { enabled, kind } : { enabled }),
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

export const remoteAccessCancelRequest = (): RemoteAccessRequest =>
  post("/custom/cancel");

export const remoteAccessTeardownRequest = (): RemoteAccessRequest =>
  post("/custom/teardown");

export function createRemoteAccessOperations<Result>(
  dispatch: (request?: RemoteAccessRequest) => Result,
) {
  return {
    load: () => dispatch(),
    start: (kind?: RemoteAccessKind) =>
      dispatch(remoteAccessStartRequest(kind)),
    stop: () => dispatch(remoteAccessStopRequest()),
    updateAutoStart: (enabled: boolean, kind?: RemoteAccessKind) =>
      dispatch(remoteAccessAutoStartRequest(enabled, kind)),
    provision: (hostname: string) =>
      dispatch(remoteAccessProvisionRequest(hostname)),
    cancel: () => dispatch(remoteAccessCancelRequest()),
    teardown: () => dispatch(remoteAccessTeardownRequest()),
  };
}

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

const remoteAccessOperations =
  createRemoteAccessOperations(requestRemoteAccess);

export const loadRemoteAccess = remoteAccessOperations.load;
export const startRemoteAccess = remoteAccessOperations.start;
export const stopRemoteAccess = remoteAccessOperations.stop;
export const updateRemoteAccessAutoStart =
  remoteAccessOperations.updateAutoStart;
export const provisionCustomRemoteAccess = remoteAccessOperations.provision;
export const cancelCustomRemoteAccess = remoteAccessOperations.cancel;
export const teardownCustomRemoteAccess = remoteAccessOperations.teardown;
