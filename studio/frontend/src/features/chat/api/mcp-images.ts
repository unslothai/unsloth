// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Backend envelope for tool images; validated so text merely mentioning it is not truncated.

export const MCP_IMAGES_MARKER = "\n__MCP_IMAGES__:";

export interface McpImage {
  data: string;
  mimeType: string;
  // Set once shortened, so the backend note reports the tool's original image count.
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

// Re-attached on replay; the backend promotes it for vision models and strips it otherwise.
export function mcpImagesEnvelope(images: McpImage[]): string {
  return MCP_IMAGES_MARKER + JSON.stringify(images);
}

// The backend keeps only the newest eight pictures, but only after parsing, so bound here too.
export const MAX_TOTAL_MCP_IMAGES = 8;
// Mirrors the backend's per-result promotion limit.
export const MAX_MODEL_IMAGES = 4;
// Mirrors LOCAL_MAX_IMAGES_PER_TURN in studio/backend/core/inference/mcp_images.py.
export const LOCAL_MAX_IMAGES_PER_TURN = 1;
// Spares past the limit: the backend counts only images that DECODE, which this side cannot tell.
export const DECODE_FAILURE_ALLOWANCE = 4;
// Total base64 cap matching the backend's per-result ceiling; the count cap alone allows ~144 MB.
export const MAX_TOTAL_MCP_IMAGE_CHARS = 12_000_000;
// Matches mcp_client.MAX_IMAGE_PAYLOAD_CHARS and mcp_images.py MAX_MCP_IMAGE_MIME_CHARS.
export const MAX_MCP_IMAGE_MIME_CHARS = 256;

export function isImageToolName(name: unknown): boolean {
  return typeof name === "string" && (name === "view_image" || name.startsWith("mcp__"));
}

interface EnvelopeCarrier {
  role?: string;
  name?: string;
  content?: unknown;
}

/**
 * Plans what each result may carry, oldest batch first. With localMarkers a batch is charged
 * once (one picture per turn). Both envelope carriers plan through here so they cannot drift.
 */
export function planMcpImageBound(
  batches: readonly (readonly (readonly McpImage[])[])[],
  { localMarkers = false }: { localMarkers?: boolean } = {},
): McpImage[][][] {
  let budget = MAX_TOTAL_MCP_IMAGES;
  // Shared across results, bounding total candidates to MAX_TOTAL_MCP_IMAGES plus the allowance.
  let spare = DECODE_FAILURE_ALLOWANCE;
  let charsLeft = MAX_TOTAL_MCP_IMAGE_CHARS;
  const perResult = localMarkers ? LOCAL_MAX_IMAGES_PER_TURN : MAX_MODEL_IMAGES;
  const out: McpImage[][][] = batches.map((batch) => batch.map(() => []));
  // Newest first: those are the ones the backend would have kept.
  for (let b = batches.length - 1; b >= 0; b--) {
    const batch = batches[b];
    const room = Math.min(budget, perResult);
    // Charge only what the backend can promote; shared spares keep corrupt entries from stranding
    // valid older images.
    let allowance = room + spare;
    let taken = 0;
    for (let r = batch.length - 1; r >= 0; r--) {
      const images = batch[r];
      // Scan rather than slice, so smaller images behind oversized entries can still fit.
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
      const returned = Math.max(images[0]?.returned ?? 0, images.length);
      out[b][r] = keep.length > 0 ? [{ ...keep[0], returned }, ...keep.slice(1)] : [];
    }
    // Fallback candidates consume spares so they cannot evict older results.
    const charged = Math.min(taken, room);
    budget -= charged;
    spare -= taken - charged;
  }
  return out;
}

/** The bound over already-serialized history, for callers holding only the wire shape. */
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
    // Strip non-MCP envelopes like the backend so their payloads cannot bypass replay bounds.
    if (
      typeof message.name === "string" &&
      message.name &&
      !isImageToolName(message.name)
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

// For a text-only target: the backend strips these anyway, so do not re-upload the base64.
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
