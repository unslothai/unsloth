// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface MediaGeneratePhaseProgress {
  // Absent or null (sd.cpp) reads as denoise.
  phase?: string | null;
  step: number;
  total: number;
  eta_seconds?: number | null;
}

export function generatePhaseLabel(
  p: MediaGeneratePhaseProgress,
  opts: { hasAudio?: boolean; formatEta?: (seconds: number) => string } = {},
): string {
  if (p.phase === "encode") return "Encoding prompt…";
  if (p.phase === "decode") return opts.hasAudio ? "Decoding video and audio…" : "Decoding…";
  if (p.phase === "export") return "Encoding video…";
  const base = "Denoising please wait…";
  // Step 0 is still running (compile warmup included), so no 0/N is shown.
  if (p.step <= 0 || p.total <= 0) return base;
  const fmt = opts.formatEta ?? ((seconds: number) => `${Math.max(0, Math.round(seconds))}s`);
  const eta = p.eta_seconds != null ? fmt(p.eta_seconds) : "";
  // Bullet separator matches ModelLoadDescription's line split; one line truncated in the 18rem card.
  return `${base} \u2022 Step ${p.step}/${p.total}${eta ? ` · ~${eta}` : ""}`;
}

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
