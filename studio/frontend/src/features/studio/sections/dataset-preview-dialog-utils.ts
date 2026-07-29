// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type PreviewImagePayload = {
  type: "image";
  mime?: string;
  width?: number;
  height?: number;
  data?: string;
};

type CollectedPreviewImage = {
  image: PreviewImagePayload;
  sourcePath: string;
};

type PreviewTraversalEntry = {
  path: Array<string | number>;
  value: unknown;
};

export function formatCell(value: unknown): string {
  if (value == null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value) || typeof value === "object") {
    return JSON.stringify(value).slice(0, 500);
  }
  return String(value);
}

function isPreviewImagePayload(value: unknown): value is PreviewImagePayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as Record<string, unknown>;
  return (
    record.type === "image" &&
    typeof record.data === "string" &&
    record.data.length > 0
  );
}

function previewChildren(
  value: unknown,
  path: Array<string | number>,
): PreviewTraversalEntry[] {
  if (Array.isArray(value)) {
    return value.map((item, index) => ({
      path: [...path, index],
      value: item,
    }));
  }
  if (!value || typeof value !== "object") {
    return [];
  }
  return Object.entries(value as Record<string, unknown>).map(
    ([key, nested]) => ({
      path: [...path, key],
      value: nested,
    }),
  );
}

export function collectPreviewImages(value: unknown): CollectedPreviewImage[] {
  const images: CollectedPreviewImage[] = [];
  const stack: PreviewTraversalEntry[] = [{ path: [], value }];
  let steps = 0;

  while (stack.length > 0 && steps < 200) {
    steps += 1;
    const entry = stack.pop();
    if (!entry) {
      break;
    }
    const { path, value: current } = entry;
    if (isPreviewImagePayload(current)) {
      images.push({
        image: current,
        sourcePath: JSON.stringify(path),
      });
      continue;
    }

    stack.push(...previewChildren(current, path));
  }

  return images;
}
