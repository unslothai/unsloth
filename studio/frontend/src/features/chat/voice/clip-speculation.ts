// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const SPECULATE_AFTER_MS = 400;

export function utteranceSpan(
  voiced: readonly boolean[],
  notSilent: readonly boolean[],
  padFrames: number,
): { start: number; end: number } | null {
  let first = voiced.indexOf(true);
  if (first < 0) return null;
  const last = voiced.lastIndexOf(true);
  while (first > 0 && notSilent[first - 1]) first--;
  return {
    start: Math.max(0, first - padFrames),
    end: Math.min(voiced.length - 1, last + padFrames),
  };
}

export function createClipSpeculation<T extends { cancel(): void }>() {
  let spec: { key: string; result: T } | null = null;
  return {
    start(key: string, analyse: () => T): void {
      if (spec?.key === key) return;
      spec?.result.cancel();
      spec = { key, result: analyse() };
    },
    take(key: string): T | null {
      const s = spec;
      spec = null;
      if (s && s.key !== key) s.result.cancel();
      return s?.key === key ? s.result : null;
    },
    clear(): void {
      spec?.result.cancel();
      spec = null;
    },
  };
}
