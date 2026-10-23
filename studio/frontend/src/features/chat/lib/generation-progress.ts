// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LiveDecodeProgress,
  LiveGenerationProgress,
  LivePrefillProgress,
} from "../stores/chat-runtime-store";
import { formatEta } from "../utils/format-transfer";

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

/** "9,216 / 23,700 tokens · 1,326 tok/s" for the Generation details setting, counted past the
 *  cached prefix like the percent. */
export function formatPrefillDetails(
  prefill: LivePrefillProgress,
  locale?: string,
): string {
  const cached = prefill.cached ?? 0;
  const done = Math.max(0, prefill.processed - cached).toLocaleString(locale);
  const parts = [
    prefill.total
      ? `${done} / ${Math.max(0, prefill.total - cached).toLocaleString(locale)} tokens`
      : `${done} tokens`,
  ];
  // The label shows only once prefillLabelShown has a rate it trusts, so the rate is past the first sample.
  if (prefill.tokPerS !== null) {
    parts.push(`${Math.round(prefill.tokPerS).toLocaleString(locale)} tok/s`);
  }
  return parts.join(" · ");
}

const MIN_TOKENS_FOR_PREFILL_RATE = 512;
const PREFILL_LABEL_MIN_SECONDS = 2;

/** Seconds left at the rate so far, or null while llama.cpp's first sample (a few tokens in, before
 *  the first batch) is all there is, since its rate is noise. */
function prefillSecondsLeft(prefill: LivePrefillProgress): number | null {
  if (!prefill.total || prefill.tokPerS === null) return null;
  if (prefill.processed - (prefill.cached ?? 0) < MIN_TOKENS_FOR_PREFILL_RATE) return null;
  return (prefill.total - prefill.processed) / prefill.tokPerS;
}

/** Whether "Processing prompt" shows after this sample. Only a pass expected to keep it up for
 *  PREFILL_LABEL_MIN_SECONDS turns it on, so a fast prompt stays on Working... instead of flashing it,
 *  and it then stays for the rest of that pass instead of dropping as the estimate shrinks. */
export function prefillLabelShown(
  previous: LiveGenerationProgress | undefined,
  prefill: LivePrefillProgress,
): boolean {
  const samePass =
    previous?.prefill != null &&
    previous.decode === null &&
    previous.prefill.total === prefill.total &&
    previous.prefill.processed <= prefill.processed;
  if (samePass && previous.prefillShown) return true;
  const left = prefillSecondsLeft(prefill);
  return left !== null && left >= PREFILL_LABEL_MIN_SECONDS;
}

/** "12s left" at the rate so far, like the load toast; null without a trusted rate or under a second. */
export function formatPrefillTimeLeft(prefill: LivePrefillProgress): string | null {
  const left = prefillSecondsLeft(prefill);
  return left !== null && left >= 1 ? `${formatEta(left)} left` : null;
}

/** "4,306 tokens · 58.3 tok/s · draft 1,791/5,030 accepted (36%)", every number in the reader's locale. */
export function formatLiveDecode(decode: LiveDecodeProgress, locale?: string): string {
  const parts = [
    `${decode.tokens.toLocaleString(locale)} ${decode.tokens === 1 ? "token" : "tokens"}`,
  ];
  if (decode.tokPerS !== null) {
    const rate = decode.tokPerS.toLocaleString(locale, {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    });
    parts.push(`${rate} tok/s`);
  }
  if (decode.draftN && decode.draftAccepted !== null) {
    const share = Math.round((decode.draftAccepted / decode.draftN) * 100);
    parts.push(
      `draft ${decode.draftAccepted.toLocaleString(locale)}/${decode.draftN.toLocaleString(locale)} accepted (${share}%)`,
    );
  }
  return parts.join(" · ");
}

/** Percent of the part of the prompt the cache did not cover, llama.cpp's "timed progress": a
 *  follow-up in a long chat would otherwise start near 100% and crawl through the last few. */
export function prefillPercent(prefill: LivePrefillProgress): number | null {
  if (!prefill.total) return null;
  const cached = prefill.cached ?? 0;
  const todo = prefill.total - cached;
  if (todo <= 0) return 100;
  return Math.min(100, Math.max(0, Math.floor(((prefill.processed - cached) / todo) * 100)));
}
