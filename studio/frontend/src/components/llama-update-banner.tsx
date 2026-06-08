// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useLlamaUpdateCheck } from "@/hooks/use-llama-update-check";
import { toast } from "@/lib/toast";
import { AnimatePresence, motion } from "motion/react";
import { type ReactElement } from "react";

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

interface LlamaUpdateBannerProps {
  enabled?: boolean;
}

/**
 * Non-invasive "Update llama.cpp" affordance. Appears bottom-right when a newer
 * prebuilt is available, fades on its own after ~10s, and re-surfaces hourly.
 * Clicking Update swaps the prebuilt in place via POST /api/llama/update.
 */
export function LlamaUpdateBanner({
  enabled = true,
}: LlamaUpdateBannerProps): ReactElement | null {
  const { status, visible, applying, apply, dismiss } = useLlamaUpdateCheck({
    enabled,
  });

  async function handleUpdate() {
    const result = await apply();
    if (result?.ok) {
      toast.success(
        `llama.cpp updated to ${result.tag ?? "the latest build"}. Reload your model to use it.`,
      );
    } else if (result) {
      toast.error(`llama.cpp update failed: ${result.error ?? "unknown error"}`);
    }
  }

  const show = visible && status != null && (status.update_available || applying);

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
          <div className="corner-squircle relative overflow-hidden border border-border/60 bg-background/95 px-4 py-3 shadow-lg backdrop-blur-md">
            {!applying ? (
              <button
                type="button"
                onClick={dismiss}
                className="absolute top-2.5 right-2.5 flex size-5 items-center justify-center rounded-md text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
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
            ) : null}

            <div className="flex items-center gap-2 pr-5">
              <span className="text-base" aria-hidden="true">
                🦥
              </span>
              <p className="text-sm font-medium text-foreground">
                {applying ? "Updating llama.cpp..." : "New llama.cpp prebuilt"}
              </p>
            </div>
            <p className="mt-0.5 pl-7 text-xs text-muted-foreground">
              {status?.installed_tag ?? "unknown"} &rarr;{" "}
              <span className="font-medium text-foreground">
                {status?.latest_tag ?? ""}
              </span>
            </p>

            <div className="mt-2.5 pl-7">
              <Button
                size="sm"
                className="corner-squircle"
                onClick={handleUpdate}
                disabled={applying}
                data-testid="llama-update-button"
              >
                {applying ? "Updating..." : "Update llama.cpp"}
              </Button>
            </div>
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
