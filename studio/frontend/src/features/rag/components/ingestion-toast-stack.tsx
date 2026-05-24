// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { useRagStore } from "../stores/rag-store";
import { IngestionProgress } from "./ingestion-progress";

/**
 * Floating progress stack for in-flight RAG ingestion jobs.
 *
 * Watches the rag-store `jobs` map (populated by `subscribeJob`) and
 * renders one card per job that's neither completed nor errored. On
 * completion the card stays for a few seconds with a success state
 * before fading out, so users notice the run finished even if they
 * weren't watching the per-doc progress chips.
 *
 * Mounted once at the app root (`__root.tsx`) so it follows users
 * across pages.
 */

const DISMISS_DELAY_MS = 4000;

export function IngestionToastStack() {
  const jobs = useRagStore((s) => s.jobs);
  const reduced = useReducedMotion();
  // Per-job timeout handles so terminal toasts auto-dismiss.
  const [dismissedJobs, setDismissedJobs] = useState<Set<string>>(
    () => new Set(),
  );

  // Schedule auto-dismiss for jobs that have reached a terminal state.
  // Ref-tracked so `dismissedJobs` isn't a useEffect dep — the setter
  // fires *inside* the effect, and depending on its output here is a
  // recipe for update-depth loops if the scheduler ever runs faster
  // than the cleanup. We snapshot the latest dismissed set into a ref
  // and read from it inside the scheduling loop instead.
  const scheduledJobsRef = useRef<Set<string>>(new Set());
  const dismissedJobsRef = useRef<Set<string>>(dismissedJobs);
  dismissedJobsRef.current = dismissedJobs;
  useEffect(() => {
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (const [jobId, event] of Object.entries(jobs)) {
      if (
        (event.type === "complete" || event.type === "error")
        && !dismissedJobsRef.current.has(jobId)
        && !scheduledJobsRef.current.has(jobId)
      ) {
        scheduledJobsRef.current.add(jobId);
        timers.push(
          setTimeout(() => {
            setDismissedJobs((prev) => {
              if (prev.has(jobId)) return prev;
              const next = new Set(prev);
              next.add(jobId);
              return next;
            });
          }, DISMISS_DELAY_MS),
        );
      }
    }
    return () => timers.forEach(clearTimeout);
  }, [jobs]);

  const visible = Object.entries(jobs).filter(
    ([jobId]) => !dismissedJobs.has(jobId),
  );
  if (visible.length === 0) return null;

  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex w-72 flex-col gap-2">
      <AnimatePresence initial={false}>
        {visible.map(([jobId, event]) => {
          const isTerminal =
            event.type === "complete" || event.type === "error";
          const isError = event.type === "error";
          return (
            <motion.div
              key={jobId}
              layout
              initial={reduced ? false : { opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={
                reduced ? { opacity: 0 } : { opacity: 0, y: 12 }
              }
              transition={{ duration: reduced ? 0 : 0.15 }}
              className="pointer-events-auto rounded-md border border-border bg-popover px-3 py-2 shadow-md"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="truncate text-xs font-medium">
                    {isError
                      ? "Ingestion failed"
                      : event.type === "complete"
                        ? "Indexed"
                        : "Indexing document"}
                  </span>
                  <IngestionProgress jobId={jobId} className="mt-0.5" />
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Dismiss"
                  className="h-5 w-5 shrink-0 text-muted-foreground hover:text-foreground"
                  onClick={() => {
                    setDismissedJobs((prev) => {
                      const next = new Set(prev);
                      next.add(jobId);
                      return next;
                    });
                  }}
                >
                  <HugeiconsIcon icon={Cancel01Icon} size={12} />
                </Button>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
