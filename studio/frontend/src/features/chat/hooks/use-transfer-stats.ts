// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Compute rate (bytes/sec) and ETA (seconds) from a time-series of cumulative
 * ``bytes`` values, using a rolling window of recent samples.
 *
 * Shared by the chat-flow download toast, training-start overlay, and
 * model-load phase UI: all three are a counter rising monotonically from 0
 * toward ``totalBytes``, polled on an interval, regardless of HTTP download or
 * mmap page-in.
 *
 * Stability: ``stable`` stays ``false`` until at least 3 samples spanning >=3s,
 * so the UI doesn't flash wild rates during the first tick or two when the
 * denominator is ~0.
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
