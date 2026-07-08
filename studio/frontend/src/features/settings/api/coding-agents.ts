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

// Which CLIs are on PATH is environment state, not a persisted setting -- it
// can change any time the user installs something new, so this only
// de-duplicates concurrent in-flight calls (e.g. React strict-mode's double
// mount) rather than caching the result across the module's lifetime. Every
// fresh call (each time a settings panel mounts) re-checks PATH for real.
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
