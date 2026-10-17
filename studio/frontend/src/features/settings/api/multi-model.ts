// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export async function loadMultiModelEnabled(
  fallbackMessage: string,
): Promise<boolean> {
  const res = await authFetch("/api/settings/multi-model");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return Boolean((await res.json()).enabled);
}

export async function updateMultiModelEnabled(
  enabled: boolean,
  fallbackMessage: string,
): Promise<boolean> {
  const res = await authFetch("/api/settings/multi-model", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return Boolean((await res.json()).enabled);
}

export const MULTI_MODEL_MODES = ["off", "fit", "always"] as const;
export type MultiModelMode = (typeof MULTI_MODEL_MODES)[number];

export function isMultiModelMode(value: unknown): value is MultiModelMode {
  return (MULTI_MODEL_MODES as readonly unknown[]).includes(value);
}

/** R1 response {enabled, default_enabled, mode, default_mode}; pre-R1 {enabled, default_enabled}. */
export function normaliseMultiModelMode(json: unknown): MultiModelMode {
  const body = (json ?? {}) as { mode?: unknown; enabled?: unknown };
  if (isMultiModelMode(body.mode)) {
    return body.mode;
  }
  // An older backend, or a mode this build does not know: the bool still decides on or off.
  return body.enabled === true ? "fit" : "off";
}

export const keepsModelsLoaded = (mode: MultiModelMode): boolean =>
  mode !== "off";

export async function loadMultiModelMode(
  fallbackMessage: string,
): Promise<MultiModelMode> {
  const res = await authFetch("/api/settings/multi-model");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return normaliseMultiModelMode(await res.json());
}

export async function updateMultiModelMode(
  mode: MultiModelMode,
  fallbackMessage: string,
): Promise<MultiModelMode> {
  const res = await authFetch("/api/settings/multi-model", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return normaliseMultiModelMode(await res.json());
}
