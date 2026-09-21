// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type InferenceEngine = "auto" | "vllm" | "sglang";
export interface EngineStatus {
  engine: Exclude<InferenceEngine, "auto">;
  version: string;
  installed_version: string | null;
  installed: boolean;
  in_use: boolean;
  current: boolean;
  restored?: boolean;
  can_rollback: boolean;
  unsupported_reason: string | null;
  job: {
    state: string;
    phase: string | null;
    message: string;
    activity?: string;
    log?: string[];
  };
}

export function isEngineReady(engine: EngineStatus | undefined): boolean {
  return !!engine?.installed &&
    (engine.current || engine.restored === true) &&
    !engine.unsupported_reason &&
    engine.job.state !== "running";
}

export async function listEngines(): Promise<EngineStatus[]> {
  const response = await authFetch("/api/engines");
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Could not read inference engines"),
    );
  }
  return response.json();
}

export async function changeEngine(
  engine: string,
  operation: "install" | "cancel" | "remove" | "rollback",
): Promise<void> {
  const response = await authFetch(
    `/api/engines/${engine}${operation === "remove" ? "" : `/${operation}`}`,
    {
      method: operation === "remove" ? "DELETE" : "POST",
    },
  );
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Could not change inference engine"),
    );
  }
  window.dispatchEvent(new Event("studio-engines-changed"));
}
