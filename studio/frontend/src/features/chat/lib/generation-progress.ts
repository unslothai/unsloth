// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LiveDecodeProgress,
  LivePrefillProgress,
} from "../stores/chat-runtime-store";

/** A `generation_progress` frame from the GGUF chat stream, read into store shape. */
export type GenerationProgressUpdate =
  | { phase: "prefill"; prefill: LivePrefillProgress }
  | { phase: "decode"; decode: LiveDecodeProgress };

function count(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.floor(value)
    : null;
}

function rate(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : null;
}

/** Null for anything malformed, so a bad frame is dropped instead of drawing NaN. */
export function readGenerationProgress(
  frame: unknown,
): GenerationProgressUpdate | null {
  if (frame === null || typeof frame !== "object") return null;
  const f = frame as Record<string, unknown>;
  if (f.phase === "prefill") {
    const processed = count(f.processed);
    if (processed === null) return null;
    const total = count(f.total);
    return {
      phase: "prefill",
      prefill: {
        processed: total ? Math.min(processed, total) : processed,
        total: total || null,
        cached: count(f.cached),
        tokPerS: rate(f.tok_per_s),
      },
    };
  }
  if (f.phase === "decode") {
    const tokens = count(f.tokens);
    if (!tokens) return null;
    const draftN = count(f.draft_n);
    const draftAccepted = count(f.draft_accepted);
    return {
      phase: "decode",
      decode: {
        tokens,
        tokPerS: rate(f.tok_per_s),
        draftN: draftN || null,
        draftAccepted:
          draftN && draftAccepted !== null ? Math.min(draftAccepted, draftN) : null,
      },
    };
  }
  return null;
}

/** "9,216 / 23,700 tokens · 1,326 tok/s" for the Generation details setting. */
export function formatPrefillDetails(
  prefill: LivePrefillProgress,
  locale?: string,
): string {
  const parts = [
    prefill.total
      ? `${prefill.processed.toLocaleString(locale)} / ${prefill.total.toLocaleString(locale)} tokens`
      : `${prefill.processed.toLocaleString(locale)} tokens`,
  ];
  // llama.cpp's first sample lands a few tokens in, before the first batch, and its rate is noise.
  const fresh = prefill.processed - (prefill.cached ?? 0);
  if (prefill.tokPerS !== null && fresh >= MIN_TOKENS_FOR_PREFILL_RATE) {
    parts.push(`${Math.round(prefill.tokPerS).toLocaleString(locale)} tok/s`);
  }
  return parts.join(" · ");
}

const MIN_TOKENS_FOR_PREFILL_RATE = 512;

/** Whole percent of the prompt processed, or null before llama.cpp reports a total. */
export function prefillPercent(prefill: LivePrefillProgress): number | null {
  if (!prefill.total) return null;
  return Math.min(100, Math.max(0, Math.floor((prefill.processed / prefill.total) * 100)));
}
