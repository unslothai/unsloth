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

export function providerCompactionMatchesTarget(
  metadata: unknown,
  providerType: string | undefined,
  modelId: string | undefined,
): boolean {
  const custom = record(metadata);
  return (
    typeof providerType === "string" &&
    typeof modelId === "string" &&
    custom?.providerCompactionProviderType === providerType &&
    custom.providerCompactionModelId === modelId
  );
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
