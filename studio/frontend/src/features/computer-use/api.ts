// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const ROUTE = "/api/computer-use";

export type ComputerUseCoordinates = "auto" | "pixels" | "grid";
export type ComputerUsePermission = "screen" | "input";

/** `available` is false when this server can't control the computer it runs on. */
export type ComputerUseStatus = {
  available: boolean;
  reason: string | null;
  enabled: boolean;
  coordinates: ComputerUseCoordinates;
  platform: string;
  /** null where the operating system has no such grant. */
  permissions: Record<ComputerUsePermission, boolean | null>;
  allowedApps: string[];
  blockedApps: string[];
  stoppedChats: number;
};

function fromApi(body: unknown): ComputerUseStatus {
  const raw = (body && typeof body === "object" ? body : {}) as Record<string, unknown>;
  const grants = (raw.permissions && typeof raw.permissions === "object" ? raw.permissions : {}) as Record<
    string,
    unknown
  >;
  const grant = (value: unknown) => (typeof value === "boolean" ? value : null);
  const names = (value: unknown) =>
    Array.isArray(value) ? value.filter((app): app is string => typeof app === "string") : [];
  return {
    available: raw.available === true,
    reason: typeof raw.reason === "string" ? raw.reason : null,
    enabled: raw.enabled === true,
    coordinates: raw.coordinates === "pixels" || raw.coordinates === "grid" ? raw.coordinates : "auto",
    platform: typeof raw.platform === "string" ? raw.platform : "none",
    permissions: { screen: grant(grants.screen), input: grant(grants.input) },
    allowedApps: names(raw.allowed_apps),
    blockedApps: names(raw.blocked_apps),
    stoppedChats: typeof raw.stopped_chats === "number" && raw.stopped_chats > 0 ? raw.stopped_chats : 0,
  };
}

async function checked(res: Response, fallback: string): Promise<ComputerUseStatus> {
  if (!res.ok) throw new Error(await readFastApiError(res, fallback));
  return fromApi(await res.json());
}

/** null from a backend older than computer use. */
export async function loadComputerUseStatus(): Promise<ComputerUseStatus | null> {
  const res = await authFetch(`${ROUTE}/status`);
  if (res.status === 404) return null;
  return checked(res, "Could not read the computer use status.");
}

export async function updateComputerUseSettings(patch: {
  enabled?: boolean;
  coordinates?: ComputerUseCoordinates;
}): Promise<ComputerUseStatus> {
  const res = await authFetch(`${ROUTE}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  return checked(res, "Could not save the computer use settings.");
}

/** Raises the operating system's own prompt; only a click in Settings calls it. */
export async function requestComputerUsePermission(kind: ComputerUsePermission): Promise<ComputerUseStatus> {
  const res = await authFetch(`${ROUTE}/permissions/${kind}`, { method: "POST" });
  return checked(res, "Could not open the system permission prompt.");
}

export async function allowComputerUseApp(name: string): Promise<ComputerUseStatus> {
  const res = await authFetch(`${ROUTE}/allowed-apps`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return checked(res, "Could not add the app.");
}

export async function removeComputerUseApp(name: string): Promise<ComputerUseStatus> {
  const res = await authFetch(`${ROUTE}/allowed-apps?name=${encodeURIComponent(name)}`, { method: "DELETE" });
  return checked(res, "Could not remove the app.");
}

export async function blockComputerUseApp(name: string): Promise<ComputerUseStatus> {
  const res = await authFetch(`${ROUTE}/blocked-apps`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return checked(res, "Could not block the app.");
}

export async function unblockComputerUseApp(name: string): Promise<ComputerUseStatus> {
  const res = await authFetch(`${ROUTE}/blocked-apps?name=${encodeURIComponent(name)}`, { method: "DELETE" });
  return checked(res, "Could not remove the app.");
}

export async function resumeComputerUse(): Promise<ComputerUseStatus> {
  const res = await authFetch(`${ROUTE}/resume`, { method: "POST" });
  return checked(res, "Could not allow computer use again.");
}

export async function revokeComputerUse(): Promise<void> {
  await authFetch(`${ROUTE}/revoke`, { method: "POST" });
}
