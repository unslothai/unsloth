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

// The canvas source view keys its Streamdown on this. Streamdown memoizes a code
// fence on its node's line/column span, ignoring the text, so equal-line-count
// canvases keep the old source. Tool artifact IDs omit the code, so hash it in.
export function buildArtifactSourceKey(
  artifact: Pick<ChatArtifact, "id" | "code">,
): string {
  return `${artifact.id}:${hashArtifactCode(artifact.code)}`;
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

export type ArtifactDownloadFormat = "html" | "md" | "txt";

interface ArtifactDownloadFormatMeta {
  label: string;
  extension: string;
  mimeType: string;
}

const ARTIFACT_DOWNLOAD_FORMAT_META: Record<
  ArtifactDownloadFormat,
  ArtifactDownloadFormatMeta
> = {
  html: {
    label: "HTML",
    extension: "html",
    mimeType: "text/html;charset=utf-8",
  },
  md: {
    label: "Markdown",
    extension: "md",
    mimeType: "text/markdown;charset=utf-8",
  },
  txt: {
    label: "Plain Text",
    extension: "txt",
    mimeType: "text/plain;charset=utf-8",
  },
};

// Order the canvas download menu is rendered in.
export const ARTIFACT_DOWNLOAD_FORMATS: readonly ArtifactDownloadFormat[] = [
  "html",
  "md",
  "txt",
];

export function getArtifactDownloadFormatLabel(
  format: ArtifactDownloadFormat,
): string {
  return ARTIFACT_DOWNLOAD_FORMAT_META[format].label;
}

export function getArtifactDownloadMimeType(
  format: ArtifactDownloadFormat,
): string {
  return ARTIFACT_DOWNLOAD_FORMAT_META[format].mimeType;
}

export function getArtifactDownloadExtension(
  format: ArtifactDownloadFormat,
): string {
  return ARTIFACT_DOWNLOAD_FORMAT_META[format].extension;
}

function slugifyArtifactTitle(title: string): string {
  return title
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

export function getArtifactFilename(
  artifact: Pick<ChatArtifact, "title">,
  format: ArtifactDownloadFormat = "html",
): string {
  const slug = slugifyArtifactTitle(artifact.title);
  return `${slug || "canvas"}.${ARTIFACT_DOWNLOAD_FORMAT_META[format].extension}`;
}

// Matches the fence the source view renders, so Markdown downloads open as a
// valid, readable document instead of dumping raw unlabeled markup.
export function buildArtifactHtmlFence(source: string): string {
  const longestBacktickRun = Math.max(
    2,
    ...(source.match(/`+/g) ?? []).map((match) => match.length),
  );
  const fence = "`".repeat(longestBacktickRun + 1);
  return `${fence}html\n${source}\n${fence}`;
}

// HTML downloads keep the raw source; Markdown wraps it in a fenced code
// block; plain text keeps the raw source too, for editors with no HTML mode.
export function buildArtifactDownloadContent(
  code: string,
  format: ArtifactDownloadFormat,
): string {
  return format === "md" ? buildArtifactHtmlFence(code) : code;
}
