// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type CurrentDatePromptSettings = {
  enabled: boolean;
  defaultEnabled: boolean;
};

type ApiCurrentDatePromptSettings = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_enabled: boolean;
};

function fromApi(
  settings: ApiCurrentDatePromptSettings,
): CurrentDatePromptSettings {
  return {
    enabled: settings.enabled,
    defaultEnabled: settings.default_enabled,
  };
}

export async function loadCurrentDatePrompt(
  fallbackMessage: string,
): Promise<CurrentDatePromptSettings> {
  const res = await authFetch("/api/settings/current-date-prompt");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, fallbackMessage),
    );
  }
  return fromApi(await res.json());
}

export async function updateCurrentDatePrompt(
  enabled: boolean,
  fallbackMessage: string,
): Promise<CurrentDatePromptSettings> {
  const res = await authFetch("/api/settings/current-date-prompt", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, fallbackMessage),
    );
  }
  return fromApi(await res.json());
}
