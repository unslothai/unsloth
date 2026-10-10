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
  // Other jobs of the same plan: done before this one, or still to come.
  for (const [file, b] of Object.entries(breakdown.earlierBytes ?? {})) add(partOf(file), Math.max(0, b), Math.max(0, b));
  for (const [file, b] of Object.entries(breakdown.laterBytes ?? {})) add(partOf(file), Math.max(0, b), 0);
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
  const { fileBytes, cachedCheckpointBytes, earlierBytes, laterBytes } = value as Record<string, unknown>;
  const sizes = sizesOf(fileBytes);
  if (!sizes) return {};
  const earlier = earlierBytes === undefined ? undefined : sizesOf(earlierBytes);
  const later = laterBytes === undefined ? undefined : sizesOf(laterBytes);
  if (earlier === null || later === null) return {};
  return {
    breakdown: {
      fileBytes: sizes,
      ...(typeof cachedCheckpointBytes === "number" &&
      Number.isFinite(cachedCheckpointBytes)
        ? { cachedCheckpointBytes }
        : {}),
      ...(earlier ? { earlierBytes: earlier } : {}),
      ...(later ? { laterBytes: later } : {}),
    },
  };
}

function sizesOf(value: unknown): Record<string, number> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const sizes = Object.entries(value);
  if (!sizes.every(([, n]) => typeof n === "number" && Number.isFinite(n))) return null;
  return Object.fromEntries(sizes) as Record<string, number>;
}

/** Gives each staged job the sizes of the plan's other jobs, so every row's bar shows model, text
 *  encoder and VAE from the start: earlier jobs as full segments, later ones as empty. A checkpoint
 *  already on disk goes on the first companion as a full model segment. */
export function withPlanBreakdown<T extends { checkpoint?: boolean; fileBytes?: Record<string, number> }>(
  entries: T[],
  checkpointBytes: number | undefined,
): (T & { cachedCheckpointBytes?: number; earlierBytes?: Record<string, number>; laterBytes?: Record<string, number> })[] {
  const cachedOn =
    checkpointBytes && checkpointBytes > 0 && !entries.some((e) => e.checkpoint)
      ? entries.findIndex((e) => !e.checkpoint)
      : -1;
  const merged = (list: T[]) => Object.assign({}, ...list.map((e) => e.fileBytes ?? {})) as Record<string, number>;
  return entries.map((e, i) => {
    const earlierBytes = merged(entries.slice(0, i));
    const laterBytes = merged(entries.slice(i + 1));
    return {
      ...e,
      ...(i === cachedOn ? { cachedCheckpointBytes: checkpointBytes } : {}),
      ...(Object.keys(earlierBytes).length ? { earlierBytes } : {}),
      ...(Object.keys(laterBytes).length ? { laterBytes } : {}),
    };
  });
}
