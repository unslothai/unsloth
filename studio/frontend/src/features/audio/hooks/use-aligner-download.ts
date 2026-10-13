// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef } from "react";

import { useStagedDownload } from "@/features/hub/download-manager";
import { toast } from "@/lib/toast";

import {
  alignerDownloadEntry,
  type SttCapabilities,
} from "../transcribe-capabilities";

/** Fetches the timing aligner through the Downloads panel before a timestamped run, instead of
 *  letting the run download it with no progress or cancel. */
export function useAlignerDownload(
  caps: SttCapabilities | null,
  refreshCaps: () => void,
  timestamps: boolean,
) {
  const pendingRun = useRef<(() => void) | null>(null);
  useEffect(() => {
    // Turning Timestamps off drops the waiting run; the download itself carries on.
    if (!timestamps) pendingRun.current = null;
  }, [timestamps]);
  const { stage, staging } = useStagedDownload({
    scopeId: "audio-aligner",
    onReady: () => {
      const run = pendingRun.current;
      pendingRun.current = null;
      refreshCaps();
      run?.();
    },
    onReset: () => {
      pendingRun.current = null;
    },
    onCancelled: () => {
      if (!pendingRun.current) return;
      pendingRun.current = null;
      toast.info("Transcription not started", {
        description:
          "Timestamps need the timing aligner. Download it again, or turn Timestamps off.",
      });
    },
  });
  const runWithAligner = useCallback(
    (timestamps: boolean, run: () => void) => {
      const entry = alignerDownloadEntry(caps, timestamps);
      if (!entry) {
        // A run without timestamps supersedes one still waiting on the aligner.
        pendingRun.current = null;
        run();
        return;
      }
      pendingRun.current = run;
      stage([entry]);
    },
    [caps, stage],
  );
  return { runWithAligner, staging };
}
