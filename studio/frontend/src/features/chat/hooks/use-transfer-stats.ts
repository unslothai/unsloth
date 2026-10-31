// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Rate and ETA from cumulative byte samples; `stable` needs 3+ samples over 3s+. */

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
      // Hidden tabs clamp polling to ~1/min, which would skew the rate; drop stale samples.
      samplesRef.current.length = 0;
      setState({ rateBytesPerSecond: 0, etaSeconds: 0, stable: false });
      return;
    }
    appendSample(samplesRef.current, now, cur);
    setState(computeTransferStats(samplesRef.current, total));
  }, [bytes, totalBytes]);

  return state;
}
