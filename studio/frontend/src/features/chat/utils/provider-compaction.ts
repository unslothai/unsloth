// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ProviderCompactionContentPart } from "../types/api";

const SERVER_SIDE_BUILTIN_TOOL_NAMES = new Set([
  "web_search",
  "web_fetch",
  "code_execution",
  "image_generation",
]);

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** Opaque, stable identity for one saved connection at one endpoint. The endpoint itself stays out of message
 * metadata, while a provider edit changes the key and invalidates provider-owned compaction state. */
export function providerCompactionConnectionKey(
  providerId: unknown,
  baseUrl: unknown,
  apiType: unknown,
): string | undefined {
  if (typeof providerId !== "string" || !providerId) return undefined;
  const identity = JSON.stringify([
    providerId,
    typeof baseUrl === "string" ? baseUrl.trim() : "",
    typeof apiType === "string" && apiType ? apiType : "chat_completions",
  ]);
  let hash = 14_695_981_039_346_656_037n;
  for (let index = 0; index < identity.length; index += 1) {
    hash ^= BigInt(identity.charCodeAt(index));
    hash = BigInt.asUintN(64, hash * 1_099_511_628_211n);
  }
  return `v1:${hash.toString(16).padStart(16, "0")}`;
}

export function providerCompactionPart(
  value: unknown,
): ProviderCompactionContentPart | null {
  if (!value || typeof value !== "object") return null;
  const { content, encrypted_content } = value as Record<string, unknown>;
  const compaction: ProviderCompactionContentPart = { type: "compaction" };
  if (typeof content === "string" && content) {
    compaction.content = content;
  }
  if (typeof encrypted_content === "string" && encrypted_content) {
    compaction.encrypted_content = encrypted_content;
  }
  return compaction.content || compaction.encrypted_content ? compaction : null;
}

function providerCompactionMatchesTarget(
  metadata: unknown,
  providerType: string | undefined,
  modelId: string | undefined,
  connectionKey: string | undefined,
): boolean {
  const custom = record(metadata);
  return (
    typeof providerType === "string" &&
    typeof modelId === "string" &&
    typeof connectionKey === "string" &&
    custom?.providerCompactionProviderType === providerType &&
    custom.providerCompactionModelId === modelId &&
    custom.providerCompactionConnectionKey === connectionKey
  );
}

export function providerCompactionForTarget(
  metadata: unknown,
  providerType: string | undefined,
  modelId: string | undefined,
  connectionKey: string | undefined,
): ProviderCompactionContentPart | null {
  if (
    !providerCompactionMatchesTarget(
      metadata,
      providerType,
      modelId,
      connectionKey,
    )
  ) {
    return null;
  }
  return providerCompactionPart(record(metadata)?.providerCompaction);
}

/** Count only persisted calls that the OpenAI replay serializer will retain. */
export function providerCompactionReplayToolCallCount(
  values: readonly unknown[],
): number {
  return values.reduce<number>((count, value) => {
    const part = record(value);
    if (part?.type !== "tool-call" || part.toolName === "studio_load_skill") {
      return count;
    }

    const args = record(part.args);
    const google = record(args?.google);
    const nativePart = google?.native_part;
    const hasNativePart =
      typeof nativePart === "object" && nativePart !== null;
    const toolName =
      typeof part.toolName === "string" ? part.toolName.toLowerCase() : "";
    const isServerSideBuiltin =
      SERVER_SIDE_BUILTIN_TOOL_NAMES.has(toolName) &&
      (args?._server_tool === true || hasNativePart);

    // Provider-hosted cards without a native Gemini part serialize to no assistant call.
    // Native cards need no role=tool result; ordinary calls do.
    if (isServerSideBuiltin) return hasNativePart ? count + 1 : count;
    return part.result !== undefined && part.result !== null ? count + 1 : count;
  }, 0);
}
