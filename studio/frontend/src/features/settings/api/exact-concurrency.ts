// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/** What the user can ask for. `auto` asks and accepts a refusal; `on` asks and fails the
 *  load if the server refuses; `off` never asks. */
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
  /** The stored value, or null when nothing is stored. Not the same as a stored "off". */
  stored: ExactConcurrencySetting | null;
  /** What the next load resolves to once the environment, this store and an inherited
   *  LLAMA_EXACT_CONCURRENCY have been read. */
  effective: string;
  /** What applies when nothing is stored. */
  fallback: string;
  /** Set while UNSLOTH_LLAMA_EXACT_CONCURRENCY pins the machine, in which case saving
   *  here changes nothing until the variable goes away. */
  envOverride: string | null;
  /** What the RUNNING server does: on, off or unavailable. */
  active: string;
  reloadRequired: boolean;
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
