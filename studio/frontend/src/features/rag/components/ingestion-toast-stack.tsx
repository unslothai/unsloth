// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { useIndexProgressStore } from "../stores/index-progress-store";

/** Single aggregate indexing toast (top-right). One entry per upload batch:
 *  "Indexing documents · 2/5 · 40%" while in flight, "RAG index ready" when
 *  the last document finishes. Replaces the old one-toast-per-file stack. */

const DISMISS_DELAY_MS = 4000;

export function IngestionToastStack() {
  const entries = useIndexProgressStore((s) => s.entries);
  const clear = useIndexProgressStore((s) => s.clear);
  const cancelAll = useIndexProgressStore((s) => s.cancelAll);
  const reduced = useReducedMotion();
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [cancelling, setCancelling] = useState(false);

  const onCancel = async () => {
    setCancelling(true);
    try {
      await cancelAll();
    } finally {
      setCancelling(false);
    }
  };

  const items = Object.values(entries);
  const total = items.length;
  const done = items.filter(
    (e) => e.status === "ready" || e.status === "error",
  ).length;
  const errored = items.filter((e) => e.status === "error").length;
  const totalChunks = items.reduce((sum, e) => sum + (e.chunks || 0), 0);
  const allDone = total > 0 && done === total;
  // Overall progress: completed/errored files count as 1, in-flight files
  // contribute their fractional progress. Smooth even for a handful of files.
  const overall =
    total === 0
      ? 0
      : items.reduce(
          (sum, e) =>
            sum + (e.status === "ready" || e.status === "error" ? 1 : e.progress),
          0,
        ) / total;
  const pct = Math.round(overall * 100);

  // Auto-dismiss once the whole batch is terminal; cancel if a new upload
  // re-opens the batch (entries change back to not-all-done).
  useEffect(() => {
    if (allDone) {
      if (dismissTimerRef.current === null) {
        dismissTimerRef.current = setTimeout(() => {
          dismissTimerRef.current = null;
          clear();
        }, DISMISS_DELAY_MS);
      }
    } else if (dismissTimerRef.current !== null) {
      clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = null;
    }
    return () => {
      if (dismissTimerRef.current !== null) {
        clearTimeout(dismissTimerRef.current);
        dismissTimerRef.current = null;
      }
    };
  }, [allDone, clear]);

  if (total === 0) return null;

  const title = allDone
    ? "RAG index ready"
    : total > 1
      ? "Indexing documents"
      : "Indexing document";

  let subtitle: string;
  if (allDone) {
    const indexed = total - errored;
    subtitle =
      `${indexed} document${indexed === 1 ? "" : "s"} and ` +
      `${totalChunks} chunk${totalChunks === 1 ? "" : "s"} indexed` +
      (errored > 0 ? ` · ${errored} failed` : "");
  } else {
    // Show the file currently being worked on (1-based), not the completed
    // count — so a fresh batch reads "1/8" rather than "0/8".
    const current = Math.min(total, done + 1);
    subtitle = `${current}/${total} · ${pct}%`;
  }

  return (
    <div className="pointer-events-none fixed right-4 top-4 z-[9999] flex w-72 flex-col gap-2">
      <AnimatePresence initial={false}>
        <motion.div
          key="rag-index-aggregate"
          layout
          initial={reduced ? false : { opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={reduced ? { opacity: 0 } : { opacity: 0, y: 12 }}
          transition={{ duration: reduced ? 0 : 0.15 }}
          className="pointer-events-auto rounded-md border border-border bg-popover px-3 py-2 shadow-md"
        >
          <div className="flex items-stretch justify-between gap-2">
            <div className="flex min-w-0 flex-col gap-1.5">
              <span className="truncate text-xs font-medium">{title}</span>
              {allDone ? (
                <span className="text-xs text-muted-foreground">{subtitle}</span>
              ) : (
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
                    <span className="truncate">
                      {errored > 0 ? "Indexing (some failed)" : "Indexing"}
                    </span>
                    <span className="shrink-0 tabular-nums">{subtitle}</span>
                  </div>
                  <Progress value={pct} className="h-1" />
                </div>
              )}
            </div>
            {/* While indexing: "Cancel" stops the batch and resets the index;
                the "X" only dismisses the toast and lets indexing continue in
                the background. When done, just the dismiss X. The X stays
                top-right; Cancel sits centered down the toast's height. */}
            <div className="flex shrink-0 flex-col items-end">
              <Button
                variant="ghost"
                size="icon"
                aria-label="Dismiss"
                className="h-5 w-5 text-muted-foreground hover:text-foreground"
                onClick={() => clear()}
              >
                <HugeiconsIcon icon={Cancel01Icon} size={12} />
              </Button>
              {!allDone && (
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={cancelling}
                  className="my-auto h-6 px-2 text-xs text-muted-foreground hover:text-destructive"
                  onClick={onCancel}
                >
                  {cancelling ? "Cancelling…" : "Cancel"}
                </Button>
              )}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
