// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Compute rate (bytes/sec) and ETA (seconds) from a time-series of
 * cumulative ``bytes`` values, using a rolling window of recent samples.
 *
 * Shared between the chat-flow download toast, the training-start
 * overlay, and the model-load phase UI. All three have the same shape:
 * a counter that rises monotonically from 0 toward ``totalBytes``, polled
 * on an interval. The derived stats are identical regardless of whether
 * the bytes came from an HTTP download or an mmap page-in.
 *
 * Stability rule: ``stable`` stays ``false`` until we've observed at
 * least 3 samples spanning ≥3 seconds. That keeps the UI from flashing
 * wildly varying rates during the first tick or two when the denominator
 * is effectively zero.
 */

import { useEffect, useRef, useState } from "react";

export type TransferStats = {
  rateBytesPerSecond: number;
  etaSeconds: number;
  /**
   * False for the first few ticks (window not filled, or no forward
   * progress yet). Consumers should hide rate/ETA while unstable so the
   * UI doesn't flicker "123 GB/s" during the first tick.
   */
  stable: boolean;
};

const MIN_SAMPLES = 3;
const MIN_WINDOW_SECONDS = 3;
const MAX_WINDOW_SECONDS = 15;

export function useTransferStats(
  bytes: number | null | undefined,
  totalBytes: number | null | undefined,
): TransferStats {
  const samplesRef = useRef<{ t: number; b: number }[]>([]);
  const [state, setState] = useState<TransferStats>({
    rateBytesPerSecond: 0,
    etaSeconds: 0,
    stable: false,
  });

  useEffect(() => {
    const now = Date.now() / 1000;
    const cur = typeof bytes === "number" && Number.isFinite(bytes) ? bytes : 0;
    const total =
      typeof totalBytes === "number" && Number.isFinite(totalBytes)
        ? totalBytes
        : 0;

    // If the counter resets (e.g. user unloaded and started a new
    // download), drop the stale window.
    const samples = samplesRef.current;
    if (samples.length > 0 && cur < samples[samples.length - 1].b) {
      samples.length = 0;
    }

    samples.push({ t: now, b: cur });

    // Drop samples older than MAX_WINDOW_SECONDS; keep at least 2 so we
    // can still compute a rate when the counter hasn't moved in a while.
    const cutoff = now - MAX_WINDOW_SECONDS;
    while (samples.length > 2 && samples[0].t < cutoff) {
      samples.shift();
    }

    if (samples.length < MIN_SAMPLES) {
      setState({ rateBytesPerSecond: 0, etaSeconds: 0, stable: false });
      return;
    }

    const first = samples[0];
    const last = samples[samples.length - 1];
    const dt = last.t - first.t;
    const db = last.b - first.b;
    if (dt < MIN_WINDOW_SECONDS || db <= 0) {
      setState({ rateBytesPerSecond: 0, etaSeconds: 0, stable: false });
      return;
    }

    const rate = db / dt;
    const eta =
      total > 0 && cur < total && rate > 0 ? (total - cur) / rate : 0;

    setState({
      rateBytesPerSecond: rate,
      etaSeconds: eta,
      stable: true,
    });
  }, [bytes, totalBytes]);

  return state;
}
