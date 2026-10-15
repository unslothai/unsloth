// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NativeIntent } from "@/features/native-intents";

export interface StagedSource {
  id: string;
  name: string;
  size: number;
  modifiedMs: number;
  dedupKey: string;
  upload: File | { nativeToken: string; expiresAtMs: number };
}

// Client-side dedup key; the backend dedups authoritatively by content hash.
export function sourceSignature(entry: StagedSource): string {
  return entry.dedupKey;
}

function stagedId(): string {
  return `staged_${Math.random().toString(36).slice(2)}`;
}

export function stagedFromFile(file: File): StagedSource {
  return {
    id: stagedId(),
    name: file.name,
    size: file.size,
    modifiedMs: file.lastModified,
    dedupKey: `${file.name}|${file.size}|${file.lastModified}`,
    upload: file,
  };
}

export function stagedFromIntent(intent: NativeIntent): StagedSource {
  const { displayLabel, expiresAtMs, modifiedMs, sizeBytes, token } =
    intent.path;
  return {
    id: stagedId(),
    name: displayLabel,
    size: sizeBytes ?? 0,
    modifiedMs: modifiedMs ?? 0,
    // Absent metadata as 0 would collapse same-named files, so fall back to the unique token.
    dedupKey:
      sizeBytes == null || modifiedMs == null
        ? `token:${token}`
        : `${displayLabel}|${sizeBytes}|${modifiedMs}`,
    upload: { nativeToken: token, expiresAtMs },
  };
}

/** Treat a native path token about to lapse as already gone. */
export const EXPIRY_GRACE_MS = 30_000;

export function nativeExpiryMs(entry: StagedSource): number | null {
  return entry.upload instanceof File ? null : entry.upload.expiresAtMs;
}

export function isExpired(entry: StagedSource, now: number): boolean {
  const expiry = nativeExpiryMs(entry);
  return expiry !== null && expiry - EXPIRY_GRACE_MS <= now;
}

export function addStagedSources(
  staged: StagedSource[],
  incoming: StagedSource[],
): { next: StagedSource[]; duplicates: string[] } {
  const seen = new Set(staged.map(sourceSignature));
  const next = [...staged];
  const duplicates: string[] = [];
  for (const entry of incoming) {
    const signature = sourceSignature(entry);
    if (seen.has(signature)) {
      duplicates.push(entry.name);
      continue;
    }
    seen.add(signature);
    next.push(entry);
  }
  return { next, duplicates };
}
