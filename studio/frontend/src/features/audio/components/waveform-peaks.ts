// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Bar heights for a waveform, from decoded samples. Free of app imports so the node test runner
// can load it directly.

export const WAVEFORM_BARS = 72;

/** One height per bar in 0..1: the loudest sample in that slice across channels, scaled so the
 *  loudest bar reaches 1. Silence stays at 0, so the bars read flat rather than amplified noise. */
export function computePeaks(
  channels: readonly Float32Array[],
  bars: number = WAVEFORM_BARS,
): number[] {
  const count = Math.max(1, Math.floor(bars));
  const length = channels.reduce(
    (longest, channel) => Math.max(longest, channel.length),
    0,
  );
  const peaks = new Array<number>(count).fill(0);
  if (length === 0) return peaks;
  for (let bar = 0; bar < count; bar += 1) {
    const start = Math.floor((bar * length) / count);
    const end = Math.max(start + 1, Math.floor(((bar + 1) * length) / count));
    let peak = 0;
    for (const channel of channels) {
      const stop = Math.min(end, channel.length);
      for (let index = start; index < stop; index += 1) {
        const value = Math.abs(channel[index]);
        if (value > peak) peak = value;
      }
    }
    peaks[bar] = Number.isFinite(peak) ? peak : 0;
  }
  const loudest = Math.max(...peaks);
  // Below about -60 dBFS everywhere: treat it as silence.
  if (loudest < 0.001) return peaks.map(() => 0);
  return peaks.map((peak) => Math.min(1, peak / loudest));
}

/** Seconds as m:ss, or h:mm:ss past an hour. */
export function formatSeconds(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds))
    return "0:00";
  const whole = Math.max(0, Math.round(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const rest = String(whole % 60).padStart(2, "0");
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${rest}`
    : `${minutes}:${rest}`;
}
