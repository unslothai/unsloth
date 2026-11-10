// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import {
  type RewardPreviewItem,
  type RewardPreviewResponse,
  previewRewards,
} from "../api/rewards-api";

/** Rewards that share one reference; a preview request takes a single reference. */
export interface RewardPreviewGroup {
  rewards: RewardPreviewItem[];
  reference: string | null;
}

/**
 * Scores a sample completion (with explanations) shortly after the inputs settle. `revision`
 * stands for anything else the scores depend on, such as a library rule edited in place.
 */
export function useRewardPreview(
  groups: RewardPreviewGroup[],
  completion: string,
  revision = "",
): { result: RewardPreviewResponse | null; error: string | null } {
  const key = JSON.stringify([groups, completion, revision]);
  const [state, setState] = useState<{
    result: RewardPreviewResponse | null;
    error: string | null;
  }>({ result: null, error: null });

  useEffect(() => {
    if (groups.length === 0) {
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const parts = await Promise.all(
          groups.map((g) =>
            previewRewards(g.rewards, completion, g.reference, true),
          ),
        );
        if (!cancelled) {
          setState({
            result: {
              scores: parts.flatMap((part) => part.scores),
              total: parts.reduce((sum, part) => sum + part.total, 0),
            },
            error: null,
          });
        }
      } catch (err) {
        if (!cancelled) {
          setState({
            result: null,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }
    }, 350);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // key stands for groups, completion and revision, which are rebuilt every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return groups.length === 0 ? { result: null, error: null } : state;
}
