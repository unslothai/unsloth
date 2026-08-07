


import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  type RemoteAccessStatus,
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
