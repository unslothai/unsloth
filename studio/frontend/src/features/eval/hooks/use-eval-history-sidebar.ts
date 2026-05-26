// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { listEvalRuns, type EvalRunSummary } from "../api/eval-api";

export const EVAL_RUNS_CHANGED = "unsloth:eval-runs-changed";

export function emitEvalRunsChanged(): void {
  window.dispatchEvent(new CustomEvent(EVAL_RUNS_CHANGED));
}

export function useEvalHistorySidebarItems(enabled: boolean): {
  items: EvalRunSummary[];
} {
  const [items, setItems] = useState<EvalRunSummary[]>([]);

  useEffect(() => {
    if (!enabled) return;
    let mounted = true;
    const load = () =>
      listEvalRuns()
        .then((res) => {
          if (mounted) setItems(res.runs);
        })
        .catch(() => {});
    load();
    const onChange = () => load();
    window.addEventListener(EVAL_RUNS_CHANGED, onChange);
    return () => {
      mounted = false;
      window.removeEventListener(EVAL_RUNS_CHANGED, onChange);
    };
  }, [enabled]);

  return { items };
}
