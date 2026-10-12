// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface PlannedDownloadEntry {
  repoId: string;
  files?: string[];
  bytes: number;
  checkpoint?: boolean;
}

/** Missing file entries, including unsized entries: zero is not proof of a cache hit. */
export function additionalAssetDownloads<T extends PlannedDownloadEntry>(
  entries: readonly T[],
): T[] {
  return entries.filter((entry) => entry.checkpoint === false);
}
export function downloadBytes(
  entries: readonly PlannedDownloadEntry[],
): number {
  return entries.reduce((sum, entry) => sum + Math.max(0, entry.bytes), 0);
}
export function selectDownloadEntries<T extends PlannedDownloadEntry>(
  entries: readonly T[],
  includeAssets: boolean,
): T[] {
  return entries.filter((entry) => includeAssets || entry.checkpoint !== false);
}
export function formatDownloadBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "Size unknown";
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(1)} TB`;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${Math.ceil(bytes / 1e3)} KB`;
}
export function assetLabel(
  entry: PlannedDownloadEntry,
  fallback: string = entry.repoId.split("/").pop() || "Required files",
): string {
  const files = entry.files ?? [];
  const weights = files.filter((f) =>
    /\.(safetensors|gguf|bin|pt|pth|ckpt)$/i.test(f),
  );
  const encoder = weights.some((f) => /text_encoder|clip|t5|qwen.*vl/i.test(f));
  const decoder = weights.some((f) => /vae|decoder|codec/i.test(f));
  if (encoder && decoder) return "Encoder & decoder";
  if (encoder) return "Text encoder";
  if (decoder) return "Decoder & configuration";
  return fallback;
}

/** Preserve dependency order while presenting and transferring checkpoints first. */
export function checkpointFirst<T extends { checkpoint?: boolean }>(
  entries: readonly T[],
): T[] {
  return [
    ...entries.filter((e) => e.checkpoint !== false),
    ...entries.filter((e) => e.checkpoint === false),
  ];
}
