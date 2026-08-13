// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The auth barrel drags the login/change-password pages (~156 kB) into the
// pill window's load graph, so import the leaf module directly.
// eslint-disable-next-line no-restricted-imports
import { authFetch } from "@/features/auth/api";
import type { PillSettings } from "@/features/system-pill";

export type { PillSettings };

export type InferenceStatus = {
  active_model: string | null;
  model_identifier: string | null;
  is_gguf: boolean;
  gguf_variant: string | null;
  loading: string[];
};

let cachedSettings: PillSettings | null = null;

// Boot race: the pill can fire before the desktop auth handshake finishes;
// a 401 is "not yet", so retry briefly instead of showing an empty pill.
async function authFetchBootTolerant(path: string): Promise<Response> {
  let response = await authFetch(path);
  for (let attempt = 0; response.status === 401 && attempt < 5; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    response = await authFetch(path);
  }
  return response;
}

export async function fetchPillSettings(): Promise<PillSettings> {
  const response = await authFetchBootTolerant("/api/pill/settings");
  if (!response.ok) {
    throw new Error(`Failed to load pill settings (${response.status})`);
  }
  cachedSettings = (await response.json()) as PillSettings;
  return cachedSettings;
}

export function getCachedSettings(): PillSettings | null {
  return cachedSettings;
}

export async function fetchInferenceStatus(): Promise<InferenceStatus> {
  const response = await authFetchBootTolerant("/api/inference/status");
  if (!response.ok) {
    throw new Error(`Failed to read inference status (${response.status})`);
  }
  return (await response.json()) as InferenceStatus;
}

export async function requestModelLoad(
  modelPath: string,
  ggufVariant: string | null,
): Promise<void> {
  // Streaming response body is ignored; the caller polls /status instead.
  const response = await authFetch("/api/inference/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_path: modelPath,
      gguf_variant: ggufVariant ?? undefined,
      max_seq_length: 0,
      load_in_4bit: true,
    }),
  });
  if (!response.ok) {
    throw new Error(`Model load failed (${response.status})`);
  }
}
