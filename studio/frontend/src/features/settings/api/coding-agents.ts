// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type CodingAgentsInfo = {
  // Every agent `unsloth start` supports, in the CLI's declared order.
  agents: string[];
  // Subset of `agents` whose CLI binary was found on PATH by the backend.
  detected: string[];
};

type ApiCodingAgentsInfo = {
  agents: string[];
  detected: string[];
};

let cachedInfo: CodingAgentsInfo | null = null;
let inFlightInfo: Promise<CodingAgentsInfo> | null = null;

function fromApi(info: ApiCodingAgentsInfo): CodingAgentsInfo {
  return { agents: info.agents, detected: info.detected };
}

async function fetchCodingAgents(): Promise<CodingAgentsInfo> {
  const res = await authFetch("/api/settings/coding-agents");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load installed coding agents"),
    );
  }
  return fromApi(await res.json());
}

export async function loadCodingAgents(): Promise<CodingAgentsInfo> {
  if (cachedInfo) {
    return cachedInfo;
  }
  inFlightInfo ??= fetchCodingAgents()
    .then((info) => {
      cachedInfo = info;
      return info;
    })
    .finally(() => {
      inFlightInfo = null;
    });
  return inFlightInfo;
}
