// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { useRagStore } from "../stores/rag-store";

/**
 * Subscribe to an ingestion job's SSE events. Returns the latest event
 * for that job from the global jobs map. Pass `null` to skip.
 */
export function useIngestionEvents(jobId: string | null) {
  const event = useRagStore((s) =>
    jobId ? (s.jobs[jobId] ?? null) : null,
  );
  const subscribeJob = useRagStore((s) => s.subscribeJob);

  useEffect(() => {
    if (jobId) subscribeJob(jobId);
  }, [jobId, subscribeJob]);

  return event;
}
