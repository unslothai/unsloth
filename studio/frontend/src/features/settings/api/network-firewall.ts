// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import { SettingsRouteAbsentError } from "./settings-route-absent";

const ROUTE = "/api/settings/network-firewall";

export const NETWORK_MODES = ["open", "ask", "allowlist"] as const;
export type NetworkMode = (typeof NETWORK_MODES)[number];

export type NetworkFirewallLists = {
  mode: NetworkMode;
  allowedSites: string[];
  blockedSites: string[];
};

export type NetworkFirewall = NetworkFirewallLists & {
  /** "locked": the OS holds Python and Terminal to the checkpoint; "watched": they are only pointed at it. */
  level: "locked" | "watched";
};

export type NetworkConnection = {
  at: number;
  tool: string;
  host: string;
  port: number;
  verdict: "allowed" | "refused";
  sent: number;
  received: number;
  count: number;
};

type Wire = {
  mode: string;
  allowed_sites: string[];
  blocked_sites: string[];
  level: string;
};

const fromWire = (wire: Wire): NetworkFirewall => ({
  mode: NETWORK_MODES.find((mode) => mode === wire.mode) ?? "open",
  allowedSites: wire.allowed_sites,
  blockedSites: wire.blocked_sites,
  level: wire.level === "locked" ? "locked" : "watched",
});

export function parseSites(text: string): string[] {
  return text
    .split(/[\r\n,]+/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

async function send(path: string, method: string, body: unknown, fallbackMessage: string) {
  const res = await authFetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readFastApiError(res, fallbackMessage));
  return fromWire((await res.json()) as Wire);
}

export async function loadNetworkFirewall(fallbackMessage: string): Promise<NetworkFirewall> {
  const res = await authFetch(ROUTE);
  if (res.status === 404) throw new SettingsRouteAbsentError(ROUTE);
  if (!res.ok) throw new Error(await readFastApiError(res, fallbackMessage));
  return fromWire((await res.json()) as Wire);
}

export function updateNetworkFirewall(
  lists: NetworkFirewallLists,
  fallbackMessage: string,
): Promise<NetworkFirewall> {
  return send(
    ROUTE,
    "PUT",
    {
      mode: lists.mode,
      allowed_sites: lists.allowedSites,
      blocked_sites: lists.blockedSites,
    },
    fallbackMessage,
  );
}

export function addNetworkSite(
  kind: "allowed" | "blocked",
  host: string,
  fallbackMessage: string,
): Promise<NetworkFirewall> {
  return send(`${ROUTE}/site`, "POST", { kind, host }, fallbackMessage);
}

export async function loadNetworkConnections(): Promise<NetworkConnection[]> {
  const res = await authFetch(`${ROUTE}/connections`);
  if (!res.ok) return [];
  return ((await res.json()) as { connections: NetworkConnection[] }).connections;
}
