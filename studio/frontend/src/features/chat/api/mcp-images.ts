// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The envelope the backend appends to a tool result that returned images.
// Validated, so tool text that merely mentions the marker is never truncated.

export const MCP_IMAGES_MARKER = "\n__MCP_IMAGES__:";

export interface McpImage {
  data: string;
  mimeType: string;
}

export function isMcpImageArray(value: unknown): value is McpImage[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (image) =>
        typeof image === "object" &&
        image !== null &&
        typeof (image as Record<string, unknown>).data === "string" &&
        typeof (image as Record<string, unknown>).mimeType === "string",
    )
  );
}

export function splitMcpImages(result: string): {
  text: string;
  images: McpImage[];
} {
  const idx = result.lastIndexOf(MCP_IMAGES_MARKER);
  if (idx === -1) return { text: result, images: [] };
  let images: unknown;
  try {
    images = JSON.parse(result.slice(idx + MCP_IMAGES_MARKER.length));
  } catch {
    return { text: result, images: [] };
  }
  if (!isMcpImageArray(images)) return { text: result, images: [] };
  return { text: result.slice(0, idx), images };
}

// Re-attached on replay: the backend promotes it into an image turn for a vision
// model, and strips it for every other one.
export function mcpImagesEnvelope(images: McpImage[]): string {
  return MCP_IMAGES_MARKER + JSON.stringify(images);
}

// The backend keeps the newest eight pictures of a conversation. It can only do
// that after the request is parsed, so without the same bound here every past
// screenshot is uploaded again on every turn and the body grows without limit.
export const MAX_TOTAL_MCP_IMAGES = 8;
// Mirrors the backend's per-result promotion limit.
export const MAX_MODEL_IMAGES = 4;
// Spare candidates carried past that limit, because the backend's quota counts
// images that DECODE and this side cannot tell which will. Bounded, so a result
// of unreadable blobs still cannot grow the request without limit.
export const DECODE_FAILURE_ALLOWANCE = 4;

const MCP_TOOL_PREFIX = "mcp__";

interface EnvelopeCarrier {
  role?: string;
  name?: string;
  content?: unknown;
}

export function boundMcpImageEnvelopes<T extends EnvelopeCarrier>(
  messages: readonly T[],
): T[] {
  const out = messages.slice();
  let budget = MAX_TOTAL_MCP_IMAGES;
  // Newest first: those are the ones the backend would have kept.
  for (let i = out.length - 1; i >= 0; i--) {
    const message = out[i];
    if (!message || message.role !== "tool") continue;
    if (typeof message.content !== "string") continue;
    // A named non-MCP result keeps its text verbatim, exactly as the backend
    // leaves it: trimming it here would change what the model reads.
    if (
      typeof message.name === "string" &&
      message.name &&
      !message.name.startsWith(MCP_TOOL_PREFIX)
    ) {
      continue;
    }
    const { text, images } = splitMcpImages(message.content);
    if (images.length === 0) continue;
    // The backend promotes at most MAX_MODEL_IMAGES out of any one result, so a
    // result carrying more must not spend history budget on images that will be
    // dropped anyway -- that would evict older results which still had room.
    //
    // But it counts SUCCESSFUL decodes, and this side cannot decode: cutting the
    // first four entries would drop valid PNGs sitting behind formats Pillow
    // rejects, which the first turn showed and the replay would then lose. Keep
    // enough candidates for the backend to still find its quota, and let it pick.
    // The slice is the remaining budget PLUS the spare candidates, not the budget
    // alone: counting spares against it strands valid PNGs sitting behind corrupt
    // entries whenever a newer result has already taken part of the allowance.
    const room = Math.min(budget, MAX_MODEL_IMAGES);
    const keep = room > 0 ? images.slice(0, room + DECODE_FAILURE_ALLOWANCE) : [];
    // Charged for what this result can actually contribute, never for room it did
    // not use, and never for the spares -- those exist only so the backend has
    // candidates to decode and must not evict an older result on their own account.
    budget -= Math.min(keep.length, room);
    if (keep.length === images.length) continue;
    out[i] = {
      ...message,
      content: keep.length > 0 ? text + mcpImagesEnvelope(keep) : text,
    };
  }
  return out;
}
