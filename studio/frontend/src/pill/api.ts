// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Leaf import: the auth barrel pulls ~156 kB of pages into the pill bundle.
// eslint-disable-next-line no-restricted-imports
import { authFetch } from "@/features/auth/api";
import { assertCompletedPaddedBody } from "@/features/chat/api/padded-response";
import type { PillSettings } from "@/features/system-pill";
import { apiBaseReady } from "@/lib/api-base";

export type { PillSettings };

export type InferenceStatus = {
  active_model: string | null;
  model_identifier: string | null;
  is_gguf: boolean;
  gguf_variant: string | null;
  loading: string[];
};

let cachedSettings: PillSettings | null = null;

// Never fetch the ':0' placeholder: WKWebView hangs such requests forever.
export async function pillFetch(input: string, init?: RequestInit): Promise<Response> {
  await apiBaseReady();
  return authFetch(input, init);
}

async function authFetchBootTolerant(
  path: string,
  signal?: AbortSignal,
): Promise<Response> {
  let response = await pillFetch(path, { signal });
  for (let attempt = 0; response.status === 401 && attempt < 5; attempt++) {
    if (signal?.aborted) return response;
    await new Promise((resolve) => setTimeout(resolve, 1500));
    response = await pillFetch(path, { signal });
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

export async function fetchInferenceStatus(
  signal?: AbortSignal,
): Promise<InferenceStatus> {
  const response = await authFetchBootTolerant("/api/inference/status", signal);
  if (!response.ok) {
    throw new Error(`Failed to read inference status (${response.status})`);
  }
  return (await response.json()) as InferenceStatus;
}

export async function requestModelLoad(
  modelPath: string,
  ggufVariant: string | null,
  signal?: AbortSignal,
): Promise<void> {
  // The 200 is committed early by keepalive padding; only the body reports load completion (_deferred_error). /status misses llama.cpp loads.
  const response = await pillFetch("/api/inference/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_path: modelPath,
      gguf_variant: ggufVariant ?? undefined,
      max_seq_length: 0,
      load_in_4bit: true,
    }),
    signal,
  });
  const body = (await response.json().catch(() => null)) as {
    _deferred_error?: { status_code?: unknown; detail?: unknown };
  } | null;
  if (!response.ok) {
    throw new Error(`Model load failed (${response.status})`);
  }
  assertCompletedPaddedBody(body, "Model load");
  const deferred = body?._deferred_error;
  if (deferred) {
    throw new Error(
      `Model load failed (${
        typeof deferred.status_code === "number" ? deferred.status_code : 500
      })`,
    );
  }
}
