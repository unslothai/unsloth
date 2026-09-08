// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type { ChatPresetSource } from "../presets/preset-policy";
import type {
  PermissionMode,
  RagAutoInject,
  RagMode,
  RagSource,
  ReasoningEffort,
} from "../stores/chat-runtime-store";
import type { ResearchWebsitePolicy } from "../types/research";
import type {
  InferenceParams,
  PersistedInferenceParams,
} from "../types/runtime";
import {
  ChatSettingsRequestError,
  isUnderKeepaliveBudget,
} from "../utils/settings-retry";

export type { PersistedInferenceParams };

export interface PersistedChatPreset {
  name: string;
  params: PersistedInferenceParams;
  loadConfig?: Record<string, unknown>;
}

export interface PersistedChatSettings {
  inferenceParams?: PersistedInferenceParams;
  /** Last-used params per checkpoint id, replayed on model switch. */
  inferenceParamsByModel?: Record<string, PersistedInferenceParams>;
  rememberParamsPerModel?: boolean;
  customPresets?: PersistedChatPreset[];
  activePreset?: string;
  activePresetSource?: ChatPresetSource;
  autoTitle?: boolean;
  reasoningEffort?: ReasoningEffort;
  preserveThinking?: boolean;
  collapseHtmlArtifacts?: boolean;
  allowArtifactNetworkAccess?: boolean;
  /** web_search also returns image results the model can place inline. */
  searchImages?: boolean;
  autoHealToolCalls?: boolean;
  nudgeToolCalls?: boolean;
  maxToolCallsPerMessage?: number;
  toolCallTimeout?: number;
  reasoningEnabled?: boolean;
  toolsEnabled?: boolean;
  codeToolsEnabled?: boolean;
  imageToolsEnabled?: boolean;
  webFetchToolsEnabled?: boolean;
  deepResearchEnabled?: boolean;
  researchWebsitePolicy?: ResearchWebsitePolicy;
  researchModelTimeoutSeconds?: number;
  artifactsEnabled?: boolean;
  showCanvasMenuItem?: boolean;
  mcpEnabledForChat?: boolean;
  confirmToolCalls?: boolean;
  /** "full" (Full access) is session-only and never leaves the browser. */
  permissionMode?: Exclude<PermissionMode, "full">;
  ragSource?: RagSource;
  ragMode?: RagMode;
  ragTopK?: number;
  ragAutoInject?: RagAutoInject;
  ragAutoInjectMinScore?: number;
  ragOcrScanned?: boolean;
  ragCaptionFigures?: boolean;
  speculativeType?: "auto" | "ngram" | "off";
  gpuMemoryMode?: "auto" | "manual";
  expandQuantizations?: boolean;
  showAllQuantizations?: boolean;
  fitOnDeviceOnly?: boolean;
  /** Local GGUF chats: drop oldest turns instead of erroring at the window. */
  autoCompactEnabled?: boolean;
  contextPolicy?: "inherit" | "checkpoint" | "rolling";
  compactionHeadroomRatio?: number;
}

interface ChatSettingsResponse {
  settings: PersistedChatSettings;
}

interface ConditionalChatSettingsResponse extends ChatSettingsResponse {
  applied: boolean;
}

export type ChatSettingsPath = [keyof PersistedChatSettings, ...string[]];

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
    // Typed, not a bare Error: the settings queue has to tell a server that is down (keep the patch)
    // from a server that refuses this body (drop the offending fields), and that decision needs the
    // status and the detail.
    throw new ChatSettingsRequestError(
      parseErrorText(response.status, body),
      response.status,
      body && typeof body === "object" && "detail" in body
        ? (body as { detail: unknown }).detail
        : null,
    );
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
  const body = JSON.stringify(patch);
  const response = await authFetch("/api/chat/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body,
    // keepalive lets the PUT survive a tab close from the page-hidden flush, but only under the Fetch
    // standard's 64 KiB budget for in-flight keepalive bodies. Over it the request fails immediately
    // in every engine, and researchWebsitePolicy alone can carry 2000 domains of 253 characters.
    // Sending without keepalive is a chance rather than a certain failure, and on the
    // visibilitychange flush, where the page is only hidden, it simply succeeds.
    keepalive: options.keepalive ? isUnderKeepaliveBudget(body) : undefined,
  });
  const data = await parseJsonOrThrow<ChatSettingsResponse>(response);
  return data.settings;
}

export async function saveChatSettingsPatchIfCurrent(
  expected: PersistedChatSettings,
  patch: PersistedChatSettings,
  expectedAbsent: Array<keyof PersistedChatSettings> = [],
  expectedAbsentPaths: ChatSettingsPath[] = [],
): Promise<{ settings: PersistedChatSettings; applied: boolean }> {
  const response = await authFetch("/api/chat/settings/compare-and-set", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      expected,
      expectedAbsent,
      expectedAbsentPaths,
      patch,
    }),
  });
  // A backend without this route answers 404 (--api-only) or 405 (the browser
  // build's GET-only SPA catch-all). The desktop app adopts any backend above a
  // version floor, so that pairing is supported, not a bug. Report "not
  // applied" so the caller leaves the server alone.
  if (response.status === 404 || response.status === 405) {
    return { settings: expected, applied: false };
  }
  return parseJsonOrThrow<ConditionalChatSettingsResponse>(response);
}
