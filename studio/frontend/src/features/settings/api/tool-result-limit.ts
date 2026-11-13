// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const ROUTE = "/api/settings/tool-result-limit";

export const DEFAULT_TOOL_RESULT_MAX_CHARS = 16000;

// Offered in Settings > Chat; a value saved through the API or set by the env var is shown beside them.
export const TOOL_RESULT_LIMIT_CHOICES = [
  4000, 8000, 16000, 32000, 64000, 128000, 200000,
] as const;

export type ToolResultLimitSettings = {
  maxChars: number;
  defaultChars: number;
  minChars: number;
  maxAllowedChars: number;
  lockedByEnvironment: boolean;
};

type ApiToolResultLimitSettings = {
  // biome-ignore lint/style/useNamingConvention: API schema
  max_chars: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_chars?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  min_chars?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  max_allowed_chars?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  locked_by_environment?: boolean;
};

export function toolResultLimitFromApi(
  settings: ApiToolResultLimitSettings,
): ToolResultLimitSettings {
  return {
    maxChars: settings.max_chars,
    defaultChars: settings.default_chars ?? DEFAULT_TOOL_RESULT_MAX_CHARS,
    minChars: settings.min_chars ?? 2000,
    maxAllowedChars: settings.max_allowed_chars ?? 200000,
    lockedByEnvironment: settings.locked_by_environment ?? false,
  };
}

/** The presets inside the server's range, plus the current value when it is not one of them. */
export function toolResultLimitChoices(
  settings: ToolResultLimitSettings,
): number[] {
  const choices: number[] = TOOL_RESULT_LIMIT_CHOICES.filter(
    (choice) =>
      choice >= settings.minChars && choice <= settings.maxAllowedChars,
  );
  if (!choices.includes(settings.maxChars)) choices.push(settings.maxChars);
  return choices.sort((a, b) => a - b);
}

export async function loadToolResultLimit(
  fallbackMessage: string,
): Promise<ToolResultLimitSettings> {
  const res = await authFetch(ROUTE);
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return toolResultLimitFromApi(await res.json());
}

export async function updateToolResultLimit(
  maxChars: number,
  fallbackMessage: string,
): Promise<ToolResultLimitSettings> {
  const res = await authFetch(ROUTE, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: API schema
    body: JSON.stringify({ max_chars: maxChars }),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return toolResultLimitFromApi(await res.json());
}
