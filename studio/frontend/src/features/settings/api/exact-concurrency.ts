// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/** `auto` accepts a refusal; `on` fails the load on one; `off` never asks. */
export type ExactConcurrencySetting = "auto" | "off" | "on";

export const EXACT_CONCURRENCY_SETTINGS: readonly ExactConcurrencySetting[] = [
  "auto",
  "off",
  "on",
];

export function isExactConcurrencySetting(
  value: string,
): value is ExactConcurrencySetting {
  return value === "auto" || value === "off" || value === "on";
}

export interface ExactConcurrencySettings {
  /** Null when nothing is stored, which differs from a stored "off". */
  stored: ExactConcurrencySetting | null;
  effective: string;
  fallback: string;
  /** Set while UNSLOTH_LLAMA_EXACT_CONCURRENCY pins the machine; saving here then changes nothing. */
  envOverride: string | null;
  active: string;
  reloadRequired: boolean;
  /** Exact mode needs server parking; without it Auto reports unavailable and On fails the load. */
  parkingAvailable: boolean;
  parkingPrerequisite: string | null;
}

interface ApiExactConcurrencySettings {
  // biome-ignore lint/style/useNamingConvention: API schema
  exact_concurrency: string | null;
  effective: string;
  default: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  env_override: string | null;
  active: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  reload_required: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  parking_available?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  parking_prerequisite?: string | null;
}

function fromApi(value: ApiExactConcurrencySettings): ExactConcurrencySettings {
  return {
    stored:
      value.exact_concurrency !== null &&
      isExactConcurrencySetting(value.exact_concurrency)
        ? value.exact_concurrency
        : null,
    effective: value.effective,
    fallback: value.default,
    envOverride: value.env_override ?? null,
    active: value.active,
    reloadRequired: value.reload_required,
    parkingAvailable: value.parking_available ?? true,
    parkingPrerequisite: value.parking_prerequisite ?? null,
  };
}

export async function loadExactConcurrencySettings(): Promise<ExactConcurrencySettings> {
  const response = await authFetch("/api/settings/exact-concurrency");
  if (!response.ok) {
    throw new Error(
      await readFastApiError(
        response,
        "Failed to load the exact concurrency setting",
      ),
    );
  }
  return fromApi(await response.json());
}

export async function updateExactConcurrencySettings(
  setting: ExactConcurrencySetting,
): Promise<ExactConcurrencySettings> {
  const response = await authFetch("/api/settings/exact-concurrency", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: API schema
    body: JSON.stringify({ exact_concurrency: setting }),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(
        response,
        "Failed to save the exact concurrency setting",
      ),
    );
  }
  return fromApi(await response.json());
}
