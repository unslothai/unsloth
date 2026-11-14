// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type ChatMemorySettings = {
  enabled: boolean;
  text: string;
  path: string;
  maxBytes: number;
  lockedByEnvironment: boolean;
};

type ApiChatMemorySettings = {
  enabled: boolean;
  text: string;
  path: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  max_bytes: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  locked_by_environment: boolean;
};

function fromApi(settings: ApiChatMemorySettings): ChatMemorySettings {
  return {
    enabled: settings.enabled,
    text: settings.text,
    path: settings.path,
    maxBytes: settings.max_bytes,
    lockedByEnvironment: settings.locked_by_environment,
  };
}

export async function loadChatMemory(
  fallbackMessage: string,
): Promise<ChatMemorySettings> {
  const res = await authFetch("/api/settings/chat-memory");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return fromApi(await res.json());
}

export async function updateChatMemory(
  patch: { enabled?: boolean; text?: string },
  fallbackMessage: string,
): Promise<ChatMemorySettings> {
  const res = await authFetch("/api/settings/chat-memory", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallbackMessage));
  }
  return fromApi(await res.json());
}
