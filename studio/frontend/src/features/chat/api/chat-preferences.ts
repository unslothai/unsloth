// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

interface ApiChatPreferences {
  // biome-ignore lint/style/useNamingConvention: API schema
  show_model_disclaimer: boolean;
}

export interface ChatPreferences {
  showModelDisclaimer: boolean;
}

function fromApi(settings: unknown): ChatPreferences {
  const value = (settings as Partial<ApiChatPreferences> | null)
    ?.show_model_disclaimer;
  if (typeof value !== "boolean") {
    throw new Error("Invalid chat preferences response");
  }
  return { showModelDisclaimer: value };
}

async function requestPreferences(
  path: string,
  init?: RequestInit,
): Promise<ChatPreferences> {
  const res = await authFetch(path, init);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Could not load chat preferences"),
    );
  }
  return fromApi(await res.json());
}

export function loadChatPreferences(): Promise<ChatPreferences> {
  return requestPreferences("/api/settings/chat-preferences");
}

export function migrateChatPreferences(
  showModelDisclaimer: boolean | undefined,
): Promise<ChatPreferences> {
  return requestPreferences("/api/settings/chat-preferences/migrate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      showModelDisclaimer === undefined
        ? {}
        : {
            // biome-ignore lint/style/useNamingConvention: API schema
            show_model_disclaimer: showModelDisclaimer,
          },
    ),
  });
}

export function updateChatPreferences(
  showModelDisclaimer: boolean,
): Promise<ChatPreferences> {
  return requestPreferences("/api/settings/chat-preferences", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      show_model_disclaimer: showModelDisclaimer,
    }),
  });
}
