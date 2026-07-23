// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { resyncInferenceStatusAfterServerModelChange } from "@/features/chat";
import { useLlamaUpdateCheck } from "@/hooks/use-llama-update-check";
import { useShowLlamaUpdateBanner } from "@/hooks/use-llama-update-pref";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Download } from "lucide-react";
import { type ReactElement, useEffect, useRef, useState } from "react";
// Creep toward this cap between coarse backend progress updates.
const RUNNING_CAP = 0.95;

// Smooth coarse backend progress without freezing between milestones.
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
      // Guard against a first rAF timestamp before the captured start time.
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
  // false fills a shared stack; true self-anchors.
  positioned?: boolean;
}

/** Bottom-right llama.cpp update toast. */
export function LlamaUpdateBanner({
  enabled = true,
  positioned = true,
}: LlamaUpdateBannerProps): ReactElement | null {
  const showBannerPref = useShowLlamaUpdateBanner();
  // Not gated on showBannerPref: this hook instance is the app-wide listener
  // for a cross-tab reload_required resync (the settings-sheet's own instance
  // only runs during an MTP-fallback rebuild), so muting the banner must not
  // also silence that resync -- it only suppresses the UI below.
  const { status, visible, applying, apply, dismiss, snooze } =
    useLlamaUpdateCheck({
      enabled,
      onReloadRequired: resyncInferenceStatusAfterServerModelChange,
    });

  async function handleUpdate() {
    const result = await apply();
    if (result?.ok) {
      const updatedTag = result.tag ?? status?.latest_tag ?? "the latest build";
      const reloadHint = result.reloadRequired
        ? " Reload your model to use it."
        : "";
      toast.success(`llama.cpp updated to ${updatedTag}.${reloadHint}`);
    } else if (result) {
      toast.error(
        `llama.cpp update failed: ${result.error ?? "unknown error"}`,
      );
    }
  }

  const show =
    showBannerPref &&
    visible &&
    status != null &&
    (status.update_available || applying);
  const sizeBytes = status?.update_size_bytes ?? null;
  const sizeLabel =
    sizeBytes && sizeBytes > 0
      ? `${Math.round(sizeBytes / (1024 * 1024))} MB`
      : null;
  const updateProgress = status?.job.progress ?? null;
  const jobSucceeded = status?.job.state === "success";
  // Display value animates; aria uses the real progress.
  const displayProgress = useSmoothedProgress(
    applying,
    updateProgress,
    jobSucceeded,
  );

  // Avoid opacity/transform transitions; GPU layer churn can flash.
  return show ? (
    <div
      className={cn(
        positioned
          ? "fixed bottom-4 right-4 z-[9998] w-[calc(100vw-32px)] max-w-[400px]"
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
              {applying ? "Updating llama.cpp..." : "New llama.cpp update"}
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {status?.installed_tag ?? "unknown"} &rarr;{" "}
              <span className="font-medium text-foreground">
                {status?.latest_tag ?? ""}
              </span>
            </p>
            <p className="mt-1 text-[0.6875rem] text-muted-foreground/70">
              {sizeLabel ? `${sizeLabel} download · ` : ""}No restart needed
              after update
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
          <div className="mt-2 flex flex-wrap items-center justify-end gap-x-1 gap-y-2">
            <Button
              size="sm"
              variant="ghost"
              className="h-auto rounded-full px-3 py-2 text-[0.8125rem] font-medium text-foreground"
              onClick={snooze}
              data-testid="llama-update-snooze-button"
            >
              Remind me later
            </Button>
            <Button
              size="sm"
              // Align pill edge with card padding.
              className="-mr-1 h-auto rounded-full px-3.5 py-2 text-[0.8125rem]"
              onClick={handleUpdate}
              data-testid="llama-update-button"
            >
              Update
            </Button>
          </div>
        )}
      </div>
    </div>
  ) : null;
}
