// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The envelope the backend appends to a tool result that returned images.
// Validated, so tool text that merely mentions the marker is never truncated.

export const MCP_IMAGES_MARKER = "\n__MCP_IMAGES__:";

export interface McpImage {
  data: string;
  mimeType: string;
  // Set on the first entry once this side has shortened the array, so the backend's
  // note can still say how many the tool returned rather than how many were uploaded.
  returned?: number;
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
// Base64 characters of replayed envelope one request may carry, across every result
// in it. The count bound alone is not a size bound: MAX_IMAGE_PAYLOAD_CHARS lets ONE
// authentic result be 12 million characters, so twelve retained candidates could
// re-upload ~144 MB on every later turn of an image-heavy chat. Set to that same
// per-result ceiling, so a whole replayed history can never cost more to send than
// the single tool result the backend already permits.
export const MAX_TOTAL_MCP_IMAGE_CHARS = 12_000_000;

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
  // Shared, so at most MAX_TOTAL_MCP_IMAGES + DECODE_FAILURE_ALLOWANCE candidates
  // ever leave here however many results there are.
  let spare = DECODE_FAILURE_ALLOWANCE;
  let charsLeft = MAX_TOTAL_MCP_IMAGE_CHARS;
  // Newest first: those are the ones the backend would have kept.
  for (let i = out.length - 1; i >= 0; i--) {
    const message = out[i];
    if (!message || message.role !== "tool") continue;
    if (typeof message.content !== "string") continue;
    const { text, images } = splitMcpImages(message.content);
    if (images.length === 0) continue;
    // A named non-MCP result is never promoted, and the backend strips its envelope
    // regardless -- so drop it here too. Left alone it bypassed both bounds below,
    // and its base64 was re-uploaded whole on every later turn.
    if (
      typeof message.name === "string" &&
      message.name &&
      !message.name.startsWith(MCP_TOOL_PREFIX)
    ) {
      out[i] = { ...message, content: text };
      continue;
    }
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
    // The allowance is spent across the CONVERSATION, not reset per result. Per
    // result it dies with the budget: four undecodable entries in the newest result
    // charge the full room, and the next result down then sees room 0 and loses its
    // envelope entirely -- so four valid PNGs are dropped while the allowance that
    // exists for exactly that case is still untouched.
    const candidates = images.slice(0, room + spare);
    // Newest first here too, so the pictures a request gives up under the byte
    // budget are the oldest ones -- the same ones every other cap here drops.
    const keep: McpImage[] = [];
    for (const image of candidates) {
      const cost = image.data.length;
      // Skip the one that does not fit and keep looking: breaking here threw away
      // three 1MB pictures sitting behind a 5MB one, which the backend could have
      // replayed. The live-result budget already skips rather than stops.
      if (charsLeft - cost < 0) continue;
      charsLeft -= cost;
      keep.push(image);
    }
    // Charged for what this result can actually contribute, never for room it did
    // not use, and never for the spares -- those exist only so the backend has
    // candidates to decode and must not evict an older result on their own account.
    const charged = Math.min(keep.length, room);
    budget -= charged;
    spare -= keep.length - charged;
    if (keep.length === images.length) continue;
    // Carry the count the tool actually returned, or a prior bound's record of it,
    // so the backend's note does not describe this upload as the whole result.
    const returned = Math.max(images[0]?.returned ?? 0, images.length);
    const bounded = keep.length > 0 ? [{ ...keep[0], returned }, ...keep.slice(1)] : [];
    out[i] = {
      ...message,
      content: bounded.length > 0 ? text + mcpImagesEnvelope(bounded) : text,
    };
  }
  return out;
}

// For a target known not to read images. The backend strips these envelopes without
// sending a pixel, so leaving them on the wire re-uploaded up to 12 million characters
// of base64 on every text turn after a switch to a text-only model. The stored
// history keeps them, for a later switch back.
export function stripMcpImageEnvelopes<T extends EnvelopeCarrier>(
  messages: readonly T[],
): T[] {
  return messages.map((message) => {
    if (!message || message.role !== "tool") return message;
    if (typeof message.content !== "string") return message;
    const { text, images } = splitMcpImages(message.content);
    return images.length === 0 ? message : { ...message, content: text };
  });
}
