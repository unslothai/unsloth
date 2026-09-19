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
// Mirrors LOCAL_MAX_IMAGES_PER_TURN in studio/backend/core/inference/mcp_images.py: the
// marker paths (safetensors, MLX) render a tool batch as one turn carrying one picture.
export const LOCAL_MAX_IMAGES_PER_TURN = 1;
// Spare candidates carried past that limit, because the backend's quota counts
// images that DECODE and this side cannot tell which will. Bounded, so a result
// of unreadable blobs still cannot grow the request without limit.
export const DECODE_FAILURE_ALLOWANCE = 4;
// Bound total replay base64 to the backend's per-result payload ceiling.
// The count cap alone permits twelve 12 MB candidates (~144 MB per turn).
export const MAX_TOTAL_MCP_IMAGE_CHARS = 12_000_000;
// The data budget matches mcp_client.MAX_IMAGE_PAYLOAD_CHARS so accepted live
// images fit their replay. Bound metadata separately, matching mcp_images.py
// MAX_MCP_IMAGE_MIME_CHARS, to prevent oversized mimeType values bypassing it.
export const MAX_MCP_IMAGE_MIME_CHARS = 256;

const MCP_TOOL_PREFIX = "mcp__";

interface EnvelopeCarrier {
  role?: string;
  name?: string;
  content?: unknown;
}

/** The bound itself, over batches of results in document order (oldest first): what
 *  each result may still carry. A batch is the results one model turn produced in
 *  parallel; the marker paths render it as one turn carrying one picture, so with
 *  localMarkers the batch is charged once and its further candidates are decode
 *  fallbacks. The part paths take MAX_MODEL_IMAGES per result, each result its own batch.
 *
 *  Both carriers of an envelope -- the serialized OpenAI history and the run's own tool
 *  results -- plan through here, so the two never drift. */
export function planMcpImageBound(
  batches: readonly (readonly (readonly McpImage[])[])[],
  { localMarkers = false }: { localMarkers?: boolean } = {},
): McpImage[][][] {
  let budget = MAX_TOTAL_MCP_IMAGES;
  // Shared, so at most MAX_TOTAL_MCP_IMAGES + DECODE_FAILURE_ALLOWANCE candidates
  // ever leave here however many results there are.
  let spare = DECODE_FAILURE_ALLOWANCE;
  let charsLeft = MAX_TOTAL_MCP_IMAGE_CHARS;
  const perResult = localMarkers ? LOCAL_MAX_IMAGES_PER_TURN : MAX_MODEL_IMAGES;
  const out: McpImage[][][] = batches.map((batch) => batch.map(() => []));
  // Newest first: those are the ones the backend would have kept.
  for (let b = batches.length - 1; b >= 0; b--) {
    const batch = batches[b];
    const room = Math.min(budget, perResult);
    // Charge only what the backend can promote; excess entries must not evict
    // older results. Since only successful decodes count and this side cannot
    // decode, keep shared fallback candidates beyond the remaining image budget.
    // Spending or resetting spares per result can strand valid older images
    // behind corrupt entries even when decoding would leave room.
    let allowance = room + spare;
    let taken = 0;
    for (let r = batch.length - 1; r >= 0; r--) {
      const images = batch[r];
      // Scan until enough candidates fit: slicing first could discard smaller
      // images behind entries that exceed the remaining byte budget.
      const keep: McpImage[] = [];
      for (const image of images) {
        if (keep.length >= allowance) break;
        if (image.mimeType.length > MAX_MCP_IMAGE_MIME_CHARS) continue;
        const cost = image.data.length;
        // Keep looking for smaller images that fit, matching the live-result budget.
        if (charsLeft - cost < 0) continue;
        charsLeft -= cost;
        keep.push(image);
      }
      allowance -= keep.length;
      taken += keep.length;
      if (keep.length === images.length) {
        out[b][r] = images.slice();
        continue;
      }
      // Carry the count the tool actually returned, or a prior bound's record of it,
      // so the backend's note does not describe this upload as the whole result.
      const returned = Math.max(images[0]?.returned ?? 0, images.length);
      out[b][r] = keep.length > 0 ? [{ ...keep[0], returned }, ...keep.slice(1)] : [];
    }
    // Charge only the batch's usable slots. Fallback candidates consume spares
    // so they cannot independently evict older results.
    const charged = Math.min(taken, room);
    budget -= charged;
    spare -= taken - charged;
  }
  return out;
}

/** The bound over an already serialized OpenAI history. The send path bounds the
 *  run's own tool results before serializing them (chat-adapter's
 *  boundMcpImageResults), so a discarded envelope is never even built; this form
 *  serves callers that only hold the wire shape. */
export function boundMcpImageEnvelopes<T extends EnvelopeCarrier>(
  messages: readonly T[],
  { localMarkers = false }: { localMarkers?: boolean } = {},
): T[] {
  const out = messages.slice();
  type Carrier = { index: number; text: string; images: McpImage[] };
  const carriers: Carrier[] = [];
  for (let i = 0; i < out.length; i++) {
    const message = out[i];
    if (!message || message.role !== "tool") continue;
    if (typeof message.content !== "string") continue;
    const { text, images } = splitMcpImages(message.content);
    if (images.length === 0) continue;
    // Match backend provenance: strip named non-MCP envelopes so their payloads
    // cannot bypass the replay bounds and be uploaded again.
    if (
      typeof message.name === "string" &&
      message.name &&
      !message.name.startsWith(MCP_TOOL_PREFIX)
    ) {
      out[i] = { ...message, content: text };
      continue;
    }
    carriers.push({ index: i, text, images });
  }
  // Consecutive tool results are one batch on a marker target.
  const batches: Carrier[][] = [];
  for (const carrier of carriers) {
    const previous = batches[batches.length - 1];
    const last = previous?.[previous.length - 1];
    let sameBatch = localMarkers && last !== undefined;
    for (let j = (last?.index ?? 0) + 1; sameBatch && j < carrier.index; j++) {
      if (out[j]?.role !== "tool") sameBatch = false;
    }
    if (sameBatch && previous) previous.push(carrier);
    else batches.push([carrier]);
  }
  const plan = planMcpImageBound(
    batches.map((batch) => batch.map((carrier) => carrier.images)),
    { localMarkers },
  );
  batches.forEach((batch, b) =>
    batch.forEach((carrier, r) => {
      const kept = plan[b][r];
      if (kept.length === carrier.images.length) return;
      out[carrier.index] = {
        ...out[carrier.index],
        content:
          kept.length > 0
            ? carrier.text + mcpImagesEnvelope(kept)
            : carrier.text,
      };
    }),
  );
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
