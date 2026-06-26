// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ChatArtifactSource = "tool" | "fence";
export type ChatArtifactSurface = "panel" | "overlay";

export interface ChatArtifact {
  id: string;
  title: string;
  code: string;
  source: ChatArtifactSource;
  sourceMessageId?: string | null;
  sourceToolCallId?: string | null;
  threadId?: string | null;
  isStreaming?: boolean;
  createdAt: number;
}

export interface ChatArtifactInput {
  title?: string | null;
  code: string;
  source: ChatArtifactSource;
  sourceMessageId?: string | null;
  sourceToolCallId?: string | null;
  threadId?: string | null;
  isStreaming?: boolean;
}

const DEFAULT_ARTIFACT_TITLE = "HTML canvas";

export function normalizeArtifactTitle(title?: string | null): string {
  const trimmed = title?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : DEFAULT_ARTIFACT_TITLE;
}

export function hashArtifactCode(code: string): string {
  let hash = 5381;
  for (let i = 0; i < code.length; i += 1) {
    hash = ((hash << 5) + hash) ^ code.charCodeAt(i);
  }
  return (hash >>> 0).toString(36);
}

export function createArtifactId(input: ChatArtifactInput): string {
  const threadSegment = input.threadId || "no-thread";
  const messageSegment = input.sourceMessageId || "transient";
  // Backend tool call IDs (call_0, call_1, …) reset per request, so the
  // message ID is needed to scope them to a specific turn.
  const parts = [input.source, threadSegment, messageSegment];

  if (input.source === "tool" && input.sourceToolCallId) {
    parts.push(input.sourceToolCallId);
  } else {
    parts.push(hashArtifactCode(input.code));
  }

  return parts.join(":");
}

export function createChatArtifact(input: ChatArtifactInput): ChatArtifact {
  return {
    id: createArtifactId(input),
    title: normalizeArtifactTitle(input.title),
    code: input.code,
    source: input.source,
    sourceMessageId: input.sourceMessageId ?? null,
    sourceToolCallId: input.sourceToolCallId ?? null,
    threadId: input.threadId ?? null,
    isStreaming: input.isStreaming,
    createdAt: Date.now(),
  };
}

export function getArtifactFilename(
  artifact: Pick<ChatArtifact, "title">,
): string {
  const slug = artifact.title
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  return `${slug || "canvas"}.html`;
}
