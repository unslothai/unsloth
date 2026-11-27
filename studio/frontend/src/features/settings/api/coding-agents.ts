// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type CodingAgentsInfo = {
  agents: string[];
  detected: string[];
};

type ApiCodingAgentsInfo = {
  agents: string[];
  detected: string[];
};

// Dedupes in-flight calls (strict-mode double mount); PATH is environment state, not cached.
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
  inFlightInfo ??= fetchCodingAgents().finally(() => {
    inFlightInfo = null;
  });
  return inFlightInfo;
}
