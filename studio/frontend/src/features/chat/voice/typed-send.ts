// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The loop only delivers what it heard, so typed text was dropped when the session ended empty.
export function isTypedVoiceSubmit(opts: {
  box: string;
  shown: string;
  paused: boolean;
  revealing?: string | null;
}): boolean {
  const box = opts.box.trim();
  if (!box) return false;
  if (isRevealPrefix(box, opts.revealing)) return false;
  if (opts.paused) return true;
  return box !== opts.shown.trim();
}

export function isRevealPrefix(box: string, revealing: string | null | undefined): boolean {
  return revealing != null && revealing.trim().startsWith(box.trim());
}
