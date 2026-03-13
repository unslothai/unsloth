// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MAX_IMAGE_PREVIEW_BYTES = 200 * 1024;

type PreviewImagePayload = {
  type?: unknown;
  mime?: unknown;
  data?: unknown;
};

type UnknownRecord = Record<string, unknown>;

export type ImagePreviewResult =
  | { kind: "ready"; src: string }
  | { kind: "too_large"; estimatedBytes: number };

function normalizeBase64(value: string): string {
  return value.replace(/\s+/g, "");
}

function estimateBase64Bytes(base64: string): number {
  const normalized = normalizeBase64(base64);
  const padding = normalized.endsWith("==")
    ? 2
    : normalized.endsWith("=")
      ? 1
      : 0;
  return Math.max(0, Math.floor((normalized.length * 3) / 4) - padding);
}

function inferMimeFromBase64(base64: string): string | null {
  const normalized = normalizeBase64(base64);
  if (normalized.startsWith("iVBORw0KGgo")) {
    return "image/png";
  }
  if (normalized.startsWith("/9j/")) {
    return "image/jpeg";
  }
  if (normalized.startsWith("R0lGOD")) {
    return "image/gif";
  }
  if (normalized.startsWith("UklGR")) {
    return "image/webp";
  }
  return null;
}

function isLikelyRawBase64Image(value: string): boolean {
  const normalized = normalizeBase64(value);
  if (normalized.length < 64) {
    return false;
  }
  if (!/^[A-Za-z0-9+/=]+$/.test(normalized)) {
    return false;
  }
  return inferMimeFromBase64(normalized) !== null;
}

function toDataUrlFromBase64(base64: string, mime: string): string {
  return `data:${mime};base64,${normalizeBase64(base64)}`;
}

function isRecord(value: unknown): value is UnknownRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isByteArray(value: unknown): value is number[] {
  if (!Array.isArray(value) || value.length === 0) {
    return false;
  }
  return value.every(
    (item) => typeof item === "number" && Number.isInteger(item) && item >= 0 && item <= 255,
  );
}

function byteArrayToBase64(bytes: number[]): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let idx = 0; idx < bytes.length; idx += chunkSize) {
    const chunk = bytes.slice(idx, idx + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function resolveStringCandidate(
  value: unknown,
  maxBytes: number,
): ImagePreviewResult | null {
  if (typeof value !== "string") {
    return null;
  }
  return resolveImagePreviewFromString(value, maxBytes);
}

function resolveImagePreviewFromString(
  value: string,
  maxBytes: number,
): ImagePreviewResult | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return { kind: "ready", src: trimmed };
  }
  if (trimmed.startsWith("data:image/")) {
    const marker = "base64,";
    const markerIdx = trimmed.indexOf(marker);
    if (markerIdx < 0) {
      return { kind: "ready", src: trimmed };
    }
    const encoded = trimmed.slice(markerIdx + marker.length);
    const estimatedBytes = estimateBase64Bytes(encoded);
    if (estimatedBytes > maxBytes) {
      return { kind: "too_large", estimatedBytes };
    }
    return { kind: "ready", src: trimmed };
  }
  if (isLikelyRawBase64Image(trimmed)) {
    const estimatedBytes = estimateBase64Bytes(trimmed);
    if (estimatedBytes > maxBytes) {
      return { kind: "too_large", estimatedBytes };
    }
    const mime = inferMimeFromBase64(trimmed) ?? "image/png";
    return { kind: "ready", src: toDataUrlFromBase64(trimmed, mime) };
  }
  return null;
}

function resolveImagePayloadObject(value: unknown, maxBytes: number): ImagePreviewResult | null {
  if (!isRecord(value)) {
    return null;
  }
  const payload = value as PreviewImagePayload;
  if (payload.type === "image" && typeof payload.data === "string") {
    const mime = typeof payload.mime === "string" ? payload.mime : "image/jpeg";
    const estimatedBytes = estimateBase64Bytes(payload.data);
    if (estimatedBytes > maxBytes) {
      return { kind: "too_large", estimatedBytes };
    }
    return {
      kind: "ready",
      src: toDataUrlFromBase64(payload.data, mime),
    };
  }

  const imageUrl = value.image_url;
  const directImageUrl = resolveStringCandidate(imageUrl, maxBytes);
  if (directImageUrl !== null) {
    return directImageUrl;
  }
  if (isRecord(imageUrl)) {
    const nestedImageUrl = resolveStringCandidate(imageUrl.url, maxBytes);
    if (nestedImageUrl !== null) {
      return nestedImageUrl;
    }
  }

  const scalarCandidates = [
    value.url,
    value.data,
    value.bytes,
    value.base64,
    value.base64_image,
    value.image,
    value.path,
  ];
  for (const candidate of scalarCandidates) {
    const resolved = resolveStringCandidate(candidate, maxBytes);
    if (resolved !== null) {
      return resolved;
    }
  }

  if (isByteArray(value.bytes)) {
    const resolved = resolveStringCandidate(byteArrayToBase64(value.bytes), maxBytes);
    if (resolved !== null) {
      return resolved;
    }
  }

  if (isRecord(value.image)) {
    return resolveImagePayloadObject(value.image, maxBytes);
  }

  return null;
}

export function resolveImagePreview(
  value: unknown,
  maxBytes = MAX_IMAGE_PREVIEW_BYTES,
): ImagePreviewResult | null {
  const payloadPreview = resolveImagePayloadObject(value, maxBytes);
  if (payloadPreview) {
    return payloadPreview;
  }

  if (typeof value !== "string") {
    return null;
  }
  return resolveImagePreviewFromString(value, maxBytes);
}

export function isLikelyImageValue(value: unknown): boolean {
  return resolveImagePreview(value, Number.POSITIVE_INFINITY) !== null;
}
