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
  llamaBackendUnavailableKey,
  visibleLlamaBackendOptions,
} from "./llama-backend-payload";
export type {
  LlamaBackend,
  LlamaBackendJob,
  LlamaBackendOption,
  LlamaBackendStatus,
  LlamaBackendSwitchStarted,
  LlamaBackendUnavailableKey,
  LlamaEffectiveBackend,
} from "./llama-backend-payload";

/** Always refetches; with forceRefresh, maxAgeSeconds allows a server resolve at most that old. */
export async function loadLlamaBackendStatus(
  forceRefresh = false,
  maxAgeSeconds?: number,
): Promise<LlamaBackendStatus> {
  const query = new URLSearchParams();
  if (forceRefresh) {
    query.set("force_refresh", "true");
    if (maxAgeSeconds !== undefined) {
      query.set("max_age_seconds", String(maxAgeSeconds));
    }
  }
  const qs = query.toString();
  const res = await authFetch(`/api/llama/backend${qs ? `?${qs}` : ""}`);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load llama.cpp backend status"),
    );
  }
  return parseLlamaBackendStatus(await res.json());
}

/** Progress arrives through loadLlamaBackendStatus().job. */
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
