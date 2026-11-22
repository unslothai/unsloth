// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { readFastApiError } from "@/lib/format-fastapi-error";
import { authFetch } from "@/features/auth";

export type MachineState =
  | "ready"
  | "busy"
  | "rpc down"
  | "unreachable"
  | "relayed"
  | "unknown";

export interface PairedMachine {
  name: string;
  node_id: string;
  state: MachineState;
  via?: string | null;
  reason?: string | null;
  where?: string | null;
  devices?: string[];
}

export interface MachinesList {
  role: "master" | "worker" | "none";
  machines: PairedMachine[];
}

export type PairingState =
  | "waiting"
  | "approval"
  | "paired"
  | "expired"
  | "refused"
  | "failed"
  | "cancelled";

export interface Pairing {
  id: string;
  state: PairingState;
  name: string | null;
  words: string[] | null;
  node_id: string | null;
  error: string | null;
  expires_at: number;
  remote: boolean;
}

async function clusterRequest(path: string, init?: RequestInit): Promise<Response> {
  const response = await authFetch(`/api/cluster${path}`, init);
  if (!response.ok) {
    throw new Error(await readFastApiError(response, "Cluster request failed"));
  }
  return response;
}

export async function fetchMachines(check = true): Promise<MachinesList> {
  const response = await clusterRequest(`/machines?check=${check}`);
  return (await response.json()) as MachinesList;
}

export async function removeMachine(nodeId: string): Promise<void> {
  await clusterRequest(`/machines/${encodeURIComponent(nodeId)}`, { method: "DELETE" });
}

// The code comes back only from this call; later state reads never include it.
export async function startPairing(): Promise<Pairing & { code: string }> {
  const response = await clusterRequest("/pair", { method: "POST" });
  return (await response.json()) as Pairing & { code: string };
}

export async function fetchPairing(id: string): Promise<Pairing> {
  const response = await clusterRequest(`/pair/${encodeURIComponent(id)}`);
  return (await response.json()) as Pairing;
}

export async function answerPairing(
  id: string,
  action: "approve" | "refuse" | "cancel",
): Promise<Pairing> {
  const response = await clusterRequest(`/pair/${encodeURIComponent(id)}/${action}`, {
    method: "POST",
  });
  return (await response.json()) as Pairing;
}
