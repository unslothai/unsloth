export const MAX_IMAGE_PREVIEW_BYTES = 200 * 1024;

type PreviewImagePayload = {
  type?: unknown;
  mime?: unknown;
  data?: unknown;
};

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

function resolveImagePayloadObject(
  value: unknown,
  maxBytes: number,
): ImagePreviewResult | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const payload = value as PreviewImagePayload;
  if (payload.type !== "image" || typeof payload.data !== "string") {
    return null;
  }
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

export function isLikelyImageValue(value: unknown): boolean {
  return resolveImagePreview(value, Number.POSITIVE_INFINITY) !== null;
}
