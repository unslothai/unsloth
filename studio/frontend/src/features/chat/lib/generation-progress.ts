// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LiveDecodeProgress,
  LiveGenerationProgress,
  LivePrefillProgress,
  LivePrefillFit,
} from "../stores/chat-runtime-store";
import { formatEta } from "../utils/format-transfer";
import type { InterpolationValues, TranslationKey } from "@/i18n";

export type Translate = (key: TranslationKey, values?: InterpolationValues) => string;

function tokenCount(count: number, t: Translate, locale?: string): string {
  return t(count === 1 ? "generationStats.tokensOne" : "generationStats.tokens", {
    count: count.toLocaleString(locale),
  });
}

/** The newest part is a finished tool call, so the next pass has not streamed yet. Sources trail a recovered reply. */
export function betweenToolRounds(
  running: boolean,
  content: readonly { type: string; result?: unknown }[],
): boolean {
  if (!running) return false;
  for (let i = content.length - 1; i >= 0; i--) {
    const part = content[i];
    if (part.type === "source") continue;
    return part.type === "tool-call" && part.result !== undefined;
  }
  return false;
}

export type GenerationProgressUpdate =
  | { phase: "prefill"; prefill: LivePrefillProgress }
  | { phase: "decode"; decode: LiveDecodeProgress }
  | { phase: "compacting" };

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
  if (f.phase === "compacting") return { phase: "compacting" };
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
        elapsedMs: count(f.time_ms),
        fit: null,
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

/** "9,216 / 23,700 tokens · 1,326 tok/s" for the Detailed speed and loading info setting, counted past the
 *  cached prefix like the percent. */
export function formatPrefillDetails(
  prefill: LivePrefillProgress,
  t: Translate,
  locale?: string,
): string {
  const cached = prefill.cached ?? 0;
  const done = Math.max(0, prefill.processed - cached);
  const parts = [
    prefill.total
      ? t("generationStats.tokensOf", {
          done: done.toLocaleString(locale),
          total: Math.max(0, prefill.total - cached).toLocaleString(locale),
        })
      : tokenCount(done, t, locale),
  ];
  // The label shows only once prefillLabelShown has a rate it trusts, so the rate is past the first sample.
  if (prefill.tokPerS !== null) {
    parts.push(`${Math.round(prefill.tokPerS).toLocaleString(locale)} tok/s`);
  }
  return parts.join(" · ");
}

const MIN_TOKENS_FOR_PREFILL_RATE = 512;
const MIN_SAMPLES_FOR_PREFILL_FIT = 4;
const FIT_FULL_AT_READ = 0.3;
const PREFILL_LABEL_MIN_SECONDS = 2;

function samePrefillPass(
  previous: LiveGenerationProgress | undefined,
  prefill: LivePrefillProgress,
): previous is LiveGenerationProgress & { prefill: LivePrefillProgress } {
  return (
    previous?.prefill != null &&
    previous.decode === null &&
    previous.prefill.total === prefill.total &&
    previous.prefill.processed <= prefill.processed
  );
}

function timedSample(prefill: LivePrefillProgress): { done: number; elapsedMs: number } | null {
  const done = prefill.processed - (prefill.cached ?? 0);
  return done >= MIN_TOKENS_FOR_PREFILL_RATE && prefill.elapsedMs
    ? { done, elapsedMs: prefill.elapsedMs }
    : null;
}

export function withPrefillFit(
  previous: LiveGenerationProgress | undefined,
  prefill: LivePrefillProgress,
): LivePrefillProgress {
  const kept = samePrefillPass(previous, prefill) ? previous.prefill.fit : null;
  const now = timedSample(prefill);
  if (now === null || (kept !== null && now.done <= kept.done)) {
    return { ...prefill, fit: kept };
  }
  const x = now.done / 1000;
  const y = now.elapsedMs / 1000;
  const terms = [1, x, x * x, x ** 3, x ** 4, y, x * y, x * x * y];
  const sums = terms.map((term, i) => (kept?.sums[i] ?? 0) + term);
  return { ...prefill, fit: { first: kept?.first ?? now.done, done: now.done, sums } };
}

function det3(m: number[][]): number {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}

// Fit c + a*x + b*x²; cap extrapolation at the observed span so one slow batch cannot dominate.
function fittedSeconds(fit: LivePrefillFit, from: number, to: number): number | null {
  const [n, sx, sx2, sx3, sx4, sy, sxy, sx2y] = fit.sums;
  if (n < MIN_SAMPLES_FOR_PREFILL_FIT) return null;
  const normal = [
    [n, sx, sx2],
    [sx, sx2, sx3],
    [sx2, sx3, sx4],
  ];
  const d = det3(normal);
  if (d > 0) {
    const withColumn = (column: number) =>
      det3(normal.map((row, i) => row.map((v, j) => (j === column ? [sy, sxy, sx2y][i] : v))));
    const a = withColumn(1) / d;
    const b = withColumn(2) / d;
    if (b > 0 && a + 2 * b * from > 0) {
      const cap = Math.min(to, from + (from - fit.first / 1000));
      const grown = a * (cap - from) + b * (cap * cap - from * from);
      return grown + (a + 2 * b * cap) * (to - cap);
    }
  }
  const slope = (n * sxy - sx * sy) / (n * sx2 - sx * sx);
  return Number.isFinite(slope) && slope > 0 ? slope * (to - from) : null;
}

export function prefillSecondsLeft(prefill: LivePrefillProgress): number | null {
  if (!prefill.total) return null;
  const cached = prefill.cached ?? 0;
  const now = timedSample(prefill);
  const rateSoFar =
    prefill.tokPerS !== null && prefill.processed - cached >= MIN_TOKENS_FOR_PREFILL_RATE
      ? (prefill.total - prefill.processed) / prefill.tokPerS
      : null;
  if (now !== null && prefill.fit !== null) {
    const todo = prefill.total - cached;
    const fitted = fittedSeconds(prefill.fit, now.done / 1000, todo / 1000);
    if (fitted !== null) {
      if (rateSoFar === null) return fitted;
      const samples = prefill.fit.sums[0];
      const weight = Math.min(1, now.done / todo / FIT_FULL_AT_READ, (samples - 3) / 3);
      return rateSoFar + weight * (fitted - rateSoFar);
    }
  }
  return rateSoFar;
}

/** Whether "Processing prompt" shows after this sample. Only a pass expected to keep it up for
 *  PREFILL_LABEL_MIN_SECONDS turns it on, so a fast prompt stays on Working... instead of flashing it,
 *  and it then stays for the rest of that pass instead of dropping as the estimate shrinks. */
export function prefillLabelShown(
  previous: LiveGenerationProgress | undefined,
  prefill: LivePrefillProgress,
): boolean {
  if (samePrefillPass(previous, prefill) && previous.prefillShown) return true;
  const left = prefillSecondsLeft(prefill);
  return left !== null && left >= PREFILL_LABEL_MIN_SECONDS;
}

export function formatPrefillTimeLeft(prefill: LivePrefillProgress, t: Translate): string | null {
  const left = prefillSecondsLeft(prefill);
  return left !== null && left >= 1 ? t("generationStats.timeLeft", { eta: formatEta(left) }) : null;
}

/** "4,306 tokens · 58.3 tok/s · draft 1,791/5,030 accepted (36%)", every number in the reader's locale. */
export function formatLiveDecode(
  decode: LiveDecodeProgress,
  t: Translate,
  locale?: string,
): string {
  const parts = [tokenCount(decode.tokens, t, locale)];
  if (decode.tokPerS !== null) {
    const rate = decode.tokPerS.toLocaleString(locale, {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    });
    parts.push(`${rate} tok/s`);
  }
  if (decode.draftN && decode.draftAccepted !== null) {
    parts.push(
      t("generationStats.draftAccepted", {
        accepted: decode.draftAccepted.toLocaleString(locale),
        drafted: decode.draftN.toLocaleString(locale),
        percent: Math.round((decode.draftAccepted / decode.draftN) * 100),
      }),
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
