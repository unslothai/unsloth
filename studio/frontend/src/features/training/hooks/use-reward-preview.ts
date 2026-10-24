// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { previewRewards, type RewardPreviewResponse } from "../api/rewards-api";
import type { RewardPreviewGroup } from "../lib/reward-editor";

export function useRewardPreview(
  groups: RewardPreviewGroup[],
  completion: string,
  revision = "",
) {
  const key = JSON.stringify({ groups, completion, revision });
  const requestController = useRef<AbortController | null>(null);
  const [result, setResult] = useState<{
    key: string;
    signal: AbortSignal;
    data: RewardPreviewResponse | null;
    error: string | null;
    pending: boolean;
  } | null>(null);
  const current = result?.key === key && !result.signal.aborted ? result : null;
  useEffect(() => () => requestController.current?.abort(), [key]);

  const run = async () => {
    if (!groups.length || current?.pending) return;
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    const state = { key, signal: controller.signal };
    setResult({ ...state, data: null, error: null, pending: true });
    try {
      const parts = await Promise.all(
        groups.map((group) =>
          previewRewards(
            group.rewards,
            completion,
            group.reference,
            controller.signal,
            true,
          ),
        ),
      );
      if (!controller.signal.aborted) {
        setResult({
          ...state,
          data: {
            scores: parts.flatMap((part) => part.scores),
            total: parts.reduce((sum, part) => sum + part.total, 0),
          },
          error: null,
          pending: false,
        });
      }
    } catch (error) {
      if (!controller.signal.aborted)
        setResult({
          ...state,
          data: null,
          error: error instanceof Error ? error.message : String(error),
          pending: false,
        });
    }
  };

  return {
    data: current?.data ?? null,
    error: current?.error ?? null,
    pending: current?.pending ?? false,
    run,
  };
}
