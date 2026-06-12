// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useLlamaUpdateCheck } from "@/hooks/use-llama-update-check";
import { useShowLlamaUpdateBanner } from "@/hooks/use-llama-update-pref";
import { toast } from "@/lib/toast";
import { AnimatePresence, motion } from "motion/react";
import type { ReactElement } from "react";

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

interface LlamaUpdateBannerProps {
  enabled?: boolean;
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

  return (
    <AnimatePresence>
      {show ? (
        <motion.div
          initial={{ opacity: 0, y: 12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.97 }}
          transition={{ duration: 0.35, ease: EASE_OUT_QUART }}
          className="fixed bottom-4 right-4 z-[9998] w-[calc(100vw-2rem)] max-w-[340px]"
          data-testid="llama-update-banner"
        >
          <div className="relative overflow-hidden rounded-[24px] bg-white px-4 pb-[22px] pl-6 pt-5 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-[0_8px_28px_-6px_rgba(0,0,0,0.28)]">
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

            <div className="min-w-0 pr-6">
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
                    : undefined
                }
                data-testid="llama-update-progress"
              >
                {updateProgress != null && updateProgress > 0 ? (
                  <div
                    className="h-full rounded-full bg-primary transition-[width] duration-700 ease-out"
                    style={{ width: `${Math.round(updateProgress * 100)}%` }}
                  />
                ) : (
                  // No percent yet (resolving the release): sweep until the
                  // first download progress arrives.
                  <div className="loading-bar-slide h-full w-1/3 rounded-full bg-primary" />
                )}
              </div>
            ) : (
              <div className="mt-3 flex items-center gap-2">
                <Button
                  size="sm"
                  className="h-auto rounded-full px-3.5 py-2 text-[13px]"
                  onClick={handleUpdate}
                  data-testid="llama-update-button"
                >
                  Update
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-auto rounded-full px-2.5 py-2 text-[13px] text-muted-foreground hover:text-foreground"
                  onClick={snooze}
                  data-testid="llama-update-snooze-button"
                >
                  Remind me later
                </Button>
              </div>
            )}
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
