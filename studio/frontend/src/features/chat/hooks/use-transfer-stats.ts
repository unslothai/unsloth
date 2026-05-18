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

import {
  appendSample,
  computeTransferStats,
  type TransferSample,
  type TransferStats,
} from "../utils/transfer-stats";

export type { TransferStats } from "../utils/transfer-stats";

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

    appendSample(samplesRef.current, now, cur);
    setState(computeTransferStats(samplesRef.current, total));
  }, [bytes, totalBytes]);

  return state;
}
