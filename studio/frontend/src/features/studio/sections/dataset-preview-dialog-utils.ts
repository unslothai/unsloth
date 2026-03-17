// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type PreviewImagePayload = {
  type: "image";
  mime?: string;
  width?: number;
  height?: number;
  data?: string;
};

export function formatCell(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value) || typeof value === "object")
    return JSON.stringify(value).slice(0, 500);
  return String(value);
}

function isPreviewImagePayload(value: unknown): value is PreviewImagePayload {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return (
    record.type === "image" &&
    typeof record.data === "string" &&
    record.data.length > 0
  );
}

export function collectPreviewImages(value: unknown): PreviewImagePayload[] {
  const images: PreviewImagePayload[] = [];
  const stack: unknown[] = [value];
  let steps = 0;

  while (stack.length > 0 && steps < 200) {
    steps += 1;
    const current = stack.pop();
    if (isPreviewImagePayload(current)) {
      images.push(current);
      continue;
    }

    if (Array.isArray(current)) {
      for (const item of current) stack.push(item);
      continue;
    }

    if (current && typeof current === "object") {
      for (const nested of Object.values(current as Record<string, unknown>)) {
        stack.push(nested);
      }
    }
  }

  return images;
}

