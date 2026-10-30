// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { pythonRewardIsolationRefusal } from "../api/rewards-api";
import { effectiveTrainingObjective } from "../lib/rl-roles";
import { useRlWorkspaceStore } from "../stores/rl-workspace-store";
import { useTrainingConfigStore } from "../stores/training-config-store";

// Re-asked while refused, so installing bubblewrap re-enables Start without a reload.
const RECHECK_MS = 10_000;

/** The refusal Start would get for this run's Python rewards on this host, or null. */
export function usePythonRewardRefusal(): string | null {
  const { objective, rewards, isolation } = useTrainingConfigStore(
    useShallow((s) => ({
      objective: effectiveTrainingObjective(s),
      rewards: s.grpoRewards,
      isolation: s.grpoPythonRewardIsolation,
    })),
  );
  const library = useRlWorkspaceStore((s) => s.library);
  const usesPython =
    objective === "grpo" &&
    rewards.some(
      (r) =>
        library.find((l) => l.name === r.name && !l.shadowed)?.kind ===
        "python",
    );
  const checks = usesPython && isolation === "required";
  const [refusal, setRefusal] = useState<string | null>(null);

  useEffect(() => {
    if (!checks) {
      setRefusal(null);
      return;
    }
    let cancelled = false;
    let timer: number | undefined;
    const check = async () => {
      let next: string | null = null;
      try {
        next = await pythonRewardIsolationRefusal("required");
      } catch {
        // Unknown: leave Start to the backend, which refuses before the model loads.
      }
      if (cancelled) return;
      setRefusal(next);
      if (next !== null) {
        timer = window.setTimeout(check, RECHECK_MS);
      }
    };
    void check();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [checks]);

  return checks ? refusal : null;
}
