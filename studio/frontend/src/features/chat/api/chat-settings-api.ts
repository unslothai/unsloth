// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type { ChatPresetSource } from "../presets/preset-policy";
import type { ReasoningEffort } from "../stores/chat-runtime-store";
import type { InferenceParams } from "../types/runtime";

export type PersistedInferenceParams = Partial<
  Omit<InferenceParams, "checkpoint">
>;

export interface PersistedChatPreset {
  name: string;
  params: PersistedInferenceParams;
}

export interface PersistedChatSettings {
  inferenceParams?: PersistedInferenceParams;
  customPresets?: PersistedChatPreset[];
  activePreset?: string;
  activePresetSource?: ChatPresetSource;
  autoTitle?: boolean;
  reasoningEffort?: ReasoningEffort;
  preserveThinking?: boolean;
  collapseHtmlArtifacts?: boolean;
  allowArtifactNetworkAccess?: boolean;
  autoHealToolCalls?: boolean;

  referenceMemories?: boolean;
  autoSaveMemories?: boolean;
  nudgeToolCalls?: boolean;
  maxToolCallsPerMessage?: number;
  toolCallTimeout?: number;
}

interface ChatSettingsResponse {
  settings: PersistedChatSettings;
}

function parseErrorText(status: number, body: unknown): string {
  if (
    body &&
    typeof body === "object" &&
    "detail" in body &&
    typeof body.detail === "string"
  ) {
    return body.detail;
  }
  if (
    body &&
    typeof body === "object" &&
    "detail" in body &&
    body.detail != null
  ) {
    return `Request failed (${status}): ${JSON.stringify(body.detail)}`;
  }
  if (
    body &&
    typeof body === "object" &&
    "message" in body &&
    typeof body.message === "string"
  ) {
    return body.message;
  }
  return `Request failed (${status})`;
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(parseErrorText(response.status, body));
  }
  return body as T;
}

export async function getChatSettings(): Promise<PersistedChatSettings> {
  const response = await authFetch("/api/chat/settings");
  const data = await parseJsonOrThrow<ChatSettingsResponse>(response);
  return data.settings;
}

export async function saveChatSettingsPatch(
  patch: PersistedChatSettings,
  options: { keepalive?: boolean } = {},
): Promise<PersistedChatSettings> {
  const response = await authFetch("/api/chat/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
    // keepalive lets the PUT survive a tab close from the beforeunload flush.
    keepalive: options.keepalive,
  });
  const data = await parseJsonOrThrow<ChatSettingsResponse>(response);
  return data.settings;
}
