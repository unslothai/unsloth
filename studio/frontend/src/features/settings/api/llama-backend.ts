// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  type LlamaBackend,
  type LlamaBackendStatus,
  type LlamaBackendSwitchStarted,
  parseLlamaBackendStatus,
  parseLlamaBackendSwitchStarted,
} from "./llama-backend-payload";

export {
  LLAMA_BACKENDS,
  isLlamaBackend,
  llamaBackendSelectionNeedsApply,
  visibleLlamaBackendOptions,
} from "./llama-backend-payload";
export type {
  LlamaBackend,
  LlamaBackendJob,
  LlamaBackendOption,
  LlamaBackendStatus,
  LlamaBackendSwitchStarted,
  LlamaEffectiveBackend,
} from "./llama-backend-payload";

/**
 * Always refetches: the payload describes the install on disk and an in-flight
 * job, both of which change under it.
 */
export async function loadLlamaBackendStatus(
  forceRefresh = false,
): Promise<LlamaBackendStatus> {
  const res = await authFetch(
    `/api/llama/backend${forceRefresh ? "?force_refresh=true" : ""}`,
  );
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load llama.cpp backend status"),
    );
  }
  return parseLlamaBackendStatus(await res.json());
}

/** Start the switch. Progress arrives through loadLlamaBackendStatus().job. */
export async function switchLlamaBackend(
  backend: LlamaBackend,
): Promise<LlamaBackendSwitchStarted> {
  const res = await authFetch("/api/llama/backend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ backend }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to switch the llama.cpp backend"),
    );
  }
  return parseLlamaBackendSwitchStarted(await res.json());
}
