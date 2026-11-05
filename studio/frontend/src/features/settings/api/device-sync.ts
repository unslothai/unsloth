// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type ClientPreferences = {
  saved: boolean;
  updatedAt: number | null;
  interfaceScale: number | null;
  browser: Record<string, unknown> | null;
};

export type ChatBackgroundRecord = {
  saved: boolean;
  updatedAt: number | null;
  adjustments: Record<string, unknown> | null;
  image: { sha256: string; type: string; size: number } | null;
};

async function json<T>(res: Response, fallback: string): Promise<T> {
  if (!res.ok) throw new Error(await readFastApiError(res, fallback));
  return (await res.json()) as T;
}

export async function loadClientPreferences(): Promise<ClientPreferences> {
  return json(
    await authFetch("/api/settings/client-preferences"),
    "Failed to load preferences",
  );
}

export async function saveClientPreferences(
  data: Partial<Pick<ClientPreferences, "interfaceScale" | "browser">>,
): Promise<ClientPreferences> {
  return json(
    await authFetch("/api/settings/client-preferences", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }),
    "Failed to save preferences",
  );
}

export async function loadChatBackground(): Promise<ChatBackgroundRecord> {
  return json(
    await authFetch("/api/settings/chat-background"),
    "Failed to load the background",
  );
}

export async function loadChatBackgroundImage(): Promise<Blob> {
  const res = await authFetch("/api/settings/chat-background/image");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to load the background"));
  }
  return res.blob();
}

export async function saveChatBackgroundImage(
  blob: Blob,
): Promise<ChatBackgroundRecord> {
  return json(
    await authFetch("/api/settings/chat-background/image", {
      method: "PUT",
      headers: { "Content-Type": blob.type || "application/octet-stream" },
      body: blob,
    }),
    "Failed to save the background",
  );
}

export async function saveChatBackgroundAdjustments(
  adjustments: Record<string, unknown>,
): Promise<ChatBackgroundRecord> {
  return json(
    await authFetch("/api/settings/chat-background", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ adjustments }),
    }),
    "Failed to save the background",
  );
}

export async function clearChatBackground(): Promise<ChatBackgroundRecord> {
  return json(
    await authFetch("/api/settings/chat-background", { method: "DELETE" }),
    "Failed to remove the background",
  );
}
