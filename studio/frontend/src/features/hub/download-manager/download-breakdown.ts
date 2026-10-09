// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DownloadBreakdown } from "./download-manager-types";

export type DownloadPartKind = "model" | "encoder" | "vae";

export interface DownloadPart {
  kind: DownloadPartKind;
  bytes: number;
  doneBytes: number;
}

// Same patterns assetLabel reads an entry's file list with.
const ENCODER = /text_encoder|clip|t5|qwen.*vl/i;
const VAE = /vae|decoder|codec/i;
const MODEL = /transformer|unet|\.gguf$/i;

// Three parts only: the small leftovers (tokenizer, scheduler, model_index.json) count as the text encoder.
function partOf(file: string): DownloadPartKind {
  if (VAE.test(file)) return "vae";
  if (MODEL.test(file) && !ENCODER.test(file)) return "model";
  return "encoder";
}

const ORDER: DownloadPartKind[] = ["model", "encoder", "vae"];

/** Bar segments for a companion download, or null when it doesn't split into two or more parts.
 *  Files arrive one at a time in name order (`snapshot_download`, `max_workers=1`), so the job's
 *  single byte count is spread over the files in that order. On a resumed download the job also
 *  counts files already on disk, so whatever `expectedBytes` holds beyond the listed files comes off
 *  the count first. */
export function downloadParts(
  breakdown: DownloadBreakdown | undefined,
  downloadedBytes: number,
  expectedBytes = 0,
): DownloadPart[] | null {
  if (!breakdown) return null;
  const parts = new Map<DownloadPartKind, DownloadPart>();
  const add = (kind: DownloadPartKind, bytes: number, doneBytes: number) => {
    const part = parts.get(kind) ?? { kind, bytes: 0, doneBytes: 0 };
    part.bytes += bytes;
    part.doneBytes += doneBytes;
    parts.set(kind, part);
  };
  const cached = Math.max(0, breakdown.cachedCheckpointBytes ?? 0);
  if (cached > 0) add("model", cached, cached);
  const listed = Object.values(breakdown.fileBytes).reduce((n, b) => n + Math.max(0, b ?? 0), 0);
  let left = Math.max(0, downloadedBytes - Math.max(0, expectedBytes - listed));
  for (const file of Object.keys(breakdown.fileBytes).sort()) {
    const bytes = Math.max(0, breakdown.fileBytes[file] ?? 0);
    if (bytes <= 0) continue;
    const done = Math.min(bytes, left);
    left -= done;
    add(partOf(file), bytes, done);
  }
  const list = ORDER.flatMap((kind) => parts.get(kind) ?? []);
  return list.length >= 2 ? list : null;
}

export function breakdownOfPersisted(
  value: unknown,
): { breakdown: DownloadBreakdown } | Record<string, never> {
  if (!value || typeof value !== "object") return {};
  const { fileBytes, cachedCheckpointBytes } = value as Record<string, unknown>;
  if (!fileBytes || typeof fileBytes !== "object" || Array.isArray(fileBytes)) {
    return {};
  }
  const sizes = Object.entries(fileBytes);
  if (!sizes.every(([, n]) => typeof n === "number" && Number.isFinite(n))) {
    return {};
  }
  return {
    breakdown: {
      fileBytes: Object.fromEntries(sizes) as Record<string, number>,
      ...(typeof cachedCheckpointBytes === "number" &&
      Number.isFinite(cachedCheckpointBytes)
        ? { cachedCheckpointBytes }
        : {}),
    },
  };
}

/** Puts the checkpoint's size on the first companion entry, so its bar shows model, text encoder and
 *  VAE together. A checkpoint the plan still downloads is its own entry, staged first, so it's on
 *  device by the time the companions run. */
export function withCachedCheckpoint<T extends { checkpoint?: boolean }>(
  entries: T[],
  checkpointBytes: number | undefined,
): (T & { cachedCheckpointBytes?: number })[] {
  if (!checkpointBytes || checkpointBytes <= 0) return entries;
  const first = entries.findIndex((e) => !e.checkpoint);
  if (first < 0) return entries;
  return entries.map((e, i) =>
    i === first ? { ...e, cachedCheckpointBytes: checkpointBytes } : e,
  );
}
