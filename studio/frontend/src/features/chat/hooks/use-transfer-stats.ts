// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Compute rate (bytes/sec) and ETA (seconds) from a time-series of cumulative `bytes` values,
 *  using a rolling window of recent samples. Shared by the chat-flow download toast,
 *  training-start overlay and model-load phase UI: all three are a counter rising monotonically
 *  toward `totalBytes`, polled on an interval. `stable` stays false until at least 3 samples
 *  spanning >=3s, so the UI does not flash wild rates while the denominator is ~0. */

import { useEffect, useRef, useState } from "react";

import {
  type TransferSample,
  type TransferStats,
  appendSample,
  computeTransferStats,
} from "@/lib/transfer-stats";

export type { TransferStats } from "@/lib/transfer-stats";

export function useTransferStats(
  bytes: number | null | undefined,
  totalBytes: number | null | undefined,
): TransferStats {
  const samplesRef = useRef<TransferSample[]>([]);
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

    if (typeof document !== "undefined" && document.hidden) {
      // Callers poll on an interval, and a hidden tab's is clamped to about once a minute, so these gaps
      // time the poller and the estimator would read them as the burst cadence. Publishing nothing also
      // stops a stale reading outliving the transfer: this effect is keyed on `bytes`, so a counter
      // that stops moving never runs it again to correct itself.
      samplesRef.current.length = 0;
      setState({ rateBytesPerSecond: 0, etaSeconds: 0, stable: false });
      return;
    }
    appendSample(samplesRef.current, now, cur);
    setState(computeTransferStats(samplesRef.current, total));
  }, [bytes, totalBytes]);

  return state;
}
