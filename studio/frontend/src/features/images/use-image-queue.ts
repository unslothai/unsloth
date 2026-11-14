// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

import { type ImageQueueJob, listDiffusionQueue } from "./api";
import { newlyFinished, queueBusy } from "./lib/generation-queue";

const BUSY_POLL_MS = 1000;
// Idle, the page still notices jobs queued over the API.
const IDLE_POLL_MS = 5000;

export function useImageQueue(active: boolean, onFinished: (jobs: ImageQueueJob[]) => void) {
  const [jobs, setJobs] = useState<ImageQueueJob[]>([]);
  // Until the first poll answers (even a failed one), an empty list means unknown, not idle.
  const [polled, setPolled] = useState(false);
  const jobsRef = useRef<ImageQueueJob[] | null>(null);
  const finishedRef = useRef(onFinished);
  finishedRef.current = onFinished;

  const apply = useCallback((next: ImageQueueJob[]) => {
    const finished = newlyFinished(jobsRef.current, next);
    jobsRef.current = next;
    setJobs(next);
    if (finished.length) finishedRef.current(finished);
  }, []);

  // Polls and action refreshes overlap on a slow link: an answer older than one already applied is dropped.
  const issued = useRef(0);
  const applied = useRef(0);
  const refresh = useCallback(async () => {
    const id = ++issued.current;
    try {
      const jobs = await listDiffusionQueue();
      if (id < applied.current) return;
      applied.current = id;
      apply(jobs);
    } catch {
      // transient; the next poll retries
    }
    setPolled(true);
  }, [apply]);

  const busy = queueBusy(jobs);
  useEffect(() => {
    if (!active) return;
    void refresh();
    const timer = setInterval(() => void refresh(), busy ? BUSY_POLL_MS : IDLE_POLL_MS);
    return () => clearInterval(timer);
  }, [active, busy, refresh]);

  return { jobs, polled, refresh, apply };
}
