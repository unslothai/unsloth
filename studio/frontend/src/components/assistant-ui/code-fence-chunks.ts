// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const FENCE_CHUNK_LINES = 32;

export type FenceLineChunks<T> = readonly (readonly T[])[];

export function updateFenceChunks<T>(
  previous: FenceLineChunks<T> | null,
  lines: readonly T[],
): FenceLineChunks<T> {
  const count = Math.ceil(lines.length / FENCE_CHUNK_LINES);
  let next: (readonly T[])[] | null =
    previous?.length === count ? null : new Array(count);
  for (let index = 0; index < count; index += 1) {
    const start = index * FENCE_CHUNK_LINES;
    const length = Math.min(FENCE_CHUNK_LINES, lines.length - start);
    const old = previous?.[index];
    let unchanged = old?.length === length;
    if (unchanged) {
      for (let offset = 0; offset < length; offset += 1) {
        if (old![offset] !== lines[start + offset]) {
          unchanged = false;
          break;
        }
      }
    }
    if (!unchanged && next === null) next = previous!.slice();
    if (next !== null) {
      next[index] = unchanged ? old! : lines.slice(start, start + length);
    }
  }
  return next ?? previous!;
}

export type FenceBodyMemoProps<Result extends { tokens: readonly unknown[] }> = {
  isIncomplete: boolean | undefined;
  language: string | null;
  result: Result | null;
  source: string;
  windowing: boolean;
};

export function sameFenceBodyProps<Result extends { tokens: readonly unknown[] }>(
  previous: FenceBodyMemoProps<Result>,
  next: FenceBodyMemoProps<Result>,
): boolean {
  return (
    previous.result === next.result &&
    previous.language === next.language &&
    previous.isIncomplete === next.isIncomplete &&
    previous.windowing === next.windowing &&
    (previous.source === next.source ||
      (previous.result !== null && previous.result.tokens.length > 0))
  );
}
