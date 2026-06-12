// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useLlamaUpdateCheck } from "@/hooks/use-llama-update-check";
import { useShowLlamaUpdateBanner } from "@/hooks/use-llama-update-pref";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Download } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { type ReactElement, useEffect, useRef, useState } from "react";

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];
// Backend progress is coarse (5% steps, ~0.9 max) and the extract tail emits no
// signal. Creep toward this cap so the bar keeps moving rather than freezing.
const RUNNING_CAP = 0.95;

// Smoothed 0..1 bar progress: eases toward real `progress`, trickles toward a
// ceiling when idle, animates to 100% when `done`. Resets to 0 on each start.
function useSmoothedProgress(
  active: boolean,
  progress: number | null,
  done: boolean,
): number {
  const [display, setDisplay] = useState(0);
  const displayRef = useRef(0);
  const progressRef = useRef<number | null>(progress);
  const doneRef = useRef(done);
  progressRef.current = progress;
  doneRef.current = done;

  useEffect(() => {
    if (!active) {
      displayRef.current = 0;
      setDisplay(0);
      return;
    }
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      // rAF timestamps can predate the performance.now() captured above, so
      // clamp dt at 0 to keep the first frame from stepping backwards.
      const dt = Math.max(0, Math.min((now - last) / 1000, 0.1));
      last = now;
      const current = displayRef.current;
      const real = progressRef.current ?? 0;
      let target: number;
      let speed: number; // approach rate (fraction of remaining gap per second)
      if (doneRef.current) {
        target = 1;
        speed = 5;
      } else if (real > current) {
        target = real; // catch up to a freshly observed milestone
        speed = 4;
      } else {
        target = RUNNING_CAP; // no signal: creep toward the cap, never frozen
        speed = 0.3;
      }
      const cap = doneRef.current ? 1 : RUNNING_CAP;
      const next = Math.min(
        current + (target - current) * Math.min(speed * dt, 1),
        cap,
      );
      displayRef.current = next;
      setDisplay(next);
      if (doneRef.current && next > 0.999) {
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);

  return display;
}

interface LlamaUpdateBannerProps {
  enabled?: boolean;
  // false: fill the parent instead of self-anchoring, so banners can stack in a
  // shared container. true (default) keeps standalone desktop mounts working.
  positioned?: boolean;
}

/**
 * Non-invasive "Update llama.cpp" affordance. Appears bottom-right ~1s after a
 * newer prebuilt is detected and stays up until the user explicitly acts on it
 * (X, Update, or Remind me later). Clicking Update swaps the prebuilt in place
 * via POST /api/llama/update. Can be turned off entirely in Settings ->
 * General -> Notifications (on by default).
 */
export function LlamaUpdateBanner({
  enabled = true,
  positioned = true,
}: LlamaUpdateBannerProps): ReactElement | null {
  const showBannerPref = useShowLlamaUpdateBanner();
  const { status, visible, applying, apply, dismiss, snooze } =
    useLlamaUpdateCheck({
      enabled: enabled && showBannerPref,
    });

  async function handleUpdate() {
    const result = await apply();
    if (result?.ok) {
      toast.success(
        `llama.cpp updated to ${result.tag ?? "the latest build"}. Reload your model to use it.`,
      );
    } else if (result) {
      toast.error(
        `llama.cpp update failed: ${result.error ?? "unknown error"}`,
      );
    }
  }

  const show =
    visible && status != null && (status.update_available || applying);
  const updateProgress = status?.job.progress ?? null;
  const jobSucceeded = status?.job.state === "success";
  // Drives the bar so it animates continuously; aria reports the real value.
  const displayProgress = useSmoothedProgress(
    applying,
    updateProgress,
    jobSucceeded,
  );

  return (
    <AnimatePresence>
      {show ? (
        <motion.div
          initial={{ opacity: 0, y: 12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.97 }}
          transition={{ duration: 0.35, ease: EASE_OUT_QUART }}
          className={cn(
            positioned
              ? "fixed bottom-4 right-4 z-[9998] w-[calc(100vw-2rem)] max-w-[400px]"
              : "pointer-events-auto w-full",
          )}
          data-testid="llama-update-banner"
        >
          <div className="relative overflow-hidden rounded-[24px] bg-white px-5 pb-4 pt-5 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-[0_8px_28px_-6px_rgba(0,0,0,0.28)]">
            {applying ? null : (
              <button
                type="button"
                onClick={dismiss}
                className="absolute top-2.5 right-3 flex size-6 items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
                aria-label="Dismiss llama.cpp update notification"
              >
                <svg
                  aria-hidden="true"
                  width="12"
                  height="12"
                  viewBox="0 0 14 14"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M11 3L3 11M3 3l8 8"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            )}

            <div className="flex min-w-0 items-start gap-4 pr-6">
              <Download
                aria-hidden="true"
                className="mt-1 size-5 shrink-0 text-foreground"
                strokeWidth={1.75}
              />
              <div className="min-w-0">
                <p className="font-heading text-base font-medium text-foreground">
                  {applying ? "Updating llama.cpp..." : "New llama.cpp version"}
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {status?.installed_tag ?? "unknown"} &rarr;{" "}
                  <span className="font-medium text-foreground">
                    {status?.latest_tag ?? ""}
                  </span>
                </p>
              </div>
            </div>

            {applying ? (
              <div
                className="mb-1.5 mt-4 h-1 overflow-hidden rounded-full bg-muted"
                role="progressbar"
                aria-label="Updating llama.cpp"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={
                  updateProgress != null
                    ? Math.round(updateProgress * 100)
                    : Math.round(displayProgress * 100)
                }
                data-testid="llama-update-progress"
              >
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${Math.max(displayProgress * 100, 2)}%` }}
                />
              </div>
            ) : (
              <div className="mt-4 flex flex-wrap items-center justify-between gap-y-2">
                <Button
                  size="sm"
                  variant="ghost"
                  className="-ml-2 h-auto rounded-full px-2.5 py-2 text-[13px] font-medium text-foreground"
                  onClick={snooze}
                  data-testid="llama-update-snooze-button"
                >
                  Remind me later
                </Button>
                <Button
                  size="sm"
                  // ml offsets the pill's filled edge so visual gaps stay equal
                  className="ml-2.5 h-auto rounded-full px-3.5 py-2 text-[13px]"
                  onClick={handleUpdate}
                  data-testid="llama-update-button"
                >
                  Update
                </Button>
              </div>
            )}
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
