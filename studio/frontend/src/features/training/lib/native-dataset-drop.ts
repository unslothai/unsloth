// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const TRAINING_DATASET_UPLOAD_EXTENSIONS = [
  ".csv",
  ".jsonl",
  ".json",
  ".parquet",
] as const;

export const TRAINING_DOCUMENT_REDIRECT_EXTENSIONS = [
  ".pdf",
  ".docx",
  ".txt",
] as const;

const DATASET_EXTENSION_SET = new Set<string>(
  TRAINING_DATASET_UPLOAD_EXTENSIONS,
);
const DOCUMENT_EXTENSION_SET = new Set<string>(
  TRAINING_DOCUMENT_REDIRECT_EXTENSIONS,
);

export type NativeTrainingDatasetDrop =
  | { kind: "dataset"; path: string; filename: string }
  | { kind: "document"; filename: string }
  | { kind: "multiple" }
  | { kind: "unsupported" };

function extensionOf(path: string): string {
  const filename = nativePathFilename(path);
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

export function nativePathFilename(path: string): string {
  const filename = path.split(/[\\/]/).pop()?.trim() ?? "";
  const sanitized = Array.from(filename, (character) => {
    const code = character.charCodeAt(0);
    return code <= 31 || code === 127 ? " " : character;
  })
    .join("")
    .trim();
  return sanitized.slice(0, 160) || "dataset";
}

export function classifyNativeTrainingDatasetDrop(
  paths: readonly string[],
): NativeTrainingDatasetDrop {
  if (paths.length !== 1) {
    return paths.length > 1 ? { kind: "multiple" } : { kind: "unsupported" };
  }
  const path = paths[0];
  if (path === undefined) {
    return { kind: "unsupported" };
  }
  const filename = nativePathFilename(path);
  const extension = extensionOf(path);
  if (DATASET_EXTENSION_SET.has(extension)) {
    return { kind: "dataset", path, filename };
  }
  if (DOCUMENT_EXTENSION_SET.has(extension)) {
    return { kind: "document", filename };
  }
  return { kind: "unsupported" };
}

export function nativeDropPositionHitsBounds(
  position: { x: number; y: number },
  scaleFactor: number,
  bounds: { left: number; right: number; top: number; bottom: number },
): boolean {
  const scale =
    Number.isFinite(scaleFactor) && scaleFactor > 0 ? scaleFactor : 1;
  const x = position.x / scale;
  const y = position.y / scale;
  return (
    x >= bounds.left &&
    x <= bounds.right &&
    y >= bounds.top &&
    y <= bounds.bottom
  );
}
