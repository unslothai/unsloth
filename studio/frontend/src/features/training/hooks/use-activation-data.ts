// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchActivations, type ActivationData } from "../api/train-api";

type UseActivationDataOptions = {
  /** Whether training is currently running — drives polling vs. one-shot. */
  isTraining: boolean;
  /** Polling interval in ms while training is running. Default: 10 000 ms. */
  pollIntervalMs?: number;
  /** Optional output directory override passed to the endpoint. */
  outputDir?: string;
};

type UseActivationDataResult = ActivationData & {
  loading: boolean;
  error: string | null;
  /** Re-fetch immediately (e.g. when jobId changes). */
  refresh: () => void;
};

export function useActivationData({
  isTraining,
  pollIntervalMs = 10_000,
  outputDir,
}: UseActivationDataOptions): UseActivationDataResult {
  const [data, setData] = useState<ActivationData>({ metadata: null, records: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const doFetch = useCallback(async () => {
    try {
      setLoading(true);
      const result = await fetchActivations(outputDir);
      if (mountedRef.current) {
        setData(result);
        setError(null);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : "Failed to load activations");
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [outputDir]);

  // Schedule polling while training is active
  useEffect(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    // Always do an initial fetch
    void doFetch();

    if (!isTraining) return;

    const schedule = () => {
      timerRef.current = setTimeout(async () => {
        await doFetch();
        if (mountedRef.current && isTraining) {
          schedule();
        }
      }, pollIntervalMs);
    };

    schedule();

    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isTraining, pollIntervalMs, doFetch]);

  return {
    ...data,
    loading,
    error,
    refresh: doFetch,
  };
}
