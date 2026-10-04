// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The fields of an image or video generate-progress poll the phase label reads. */
export interface MediaGeneratePhaseProgress {
  // encode | denoise | decode | export (video); absent or null (sd.cpp) reads as denoise.
  phase?: string | null;
  step: number;
  total: number;
  eta_seconds?: number | null;
}

/** The text shown beside the generation spinner, the same on the Images and Video pages.
 *  ``hasAudio`` only changes the decode wording (video families that decode a soundtrack too);
 *  ``formatEta`` renders the remaining seconds (the pages pass the hub formatter). */
export function generatePhaseLabel(
  p: MediaGeneratePhaseProgress,
  opts: { hasAudio?: boolean; formatEta?: (seconds: number) => string } = {},
): string {
  if (p.phase === "encode") return "Encoding prompt…";
  if (p.phase === "decode") return opts.hasAudio ? "Decoding video and audio…" : "Decoding…";
  if (p.phase === "export") return "Encoding video…";
  const base = "Denoising please wait…";
  // Step 0 is the first step still running (compile warmup included), so show no 0/N.
  if (p.step <= 0 || p.total <= 0) return base;
  const fmt = opts.formatEta ?? ((seconds: number) => `${Math.max(0, Math.round(seconds))}s`);
  const eta = p.eta_seconds != null ? fmt(p.eta_seconds) : "";
  return `${base} Step ${p.step}/${p.total}${eta ? ` · ~${eta}` : ""}`;
}

/** Whether two polls would render the same card and preview, so the poll can skip a re-render. */
export function sameGenerateProgress(
  a: { step: number; eta_seconds?: number | null; phase?: string | null; preview_seq?: number },
  b: { step: number; eta_seconds?: number | null; phase?: string | null; preview_seq?: number },
): boolean {
  return (
    a.step === b.step &&
    a.eta_seconds === b.eta_seconds &&
    a.phase === b.phase &&
    (a.preview_seq ?? 0) === (b.preview_seq ?? 0)
  );
}
