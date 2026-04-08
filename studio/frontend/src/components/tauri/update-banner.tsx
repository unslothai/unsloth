// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import type { UpdateInfo, UpdateStatus } from "@/hooks/use-tauri-update";
import { AnimatePresence, motion } from "motion/react";

interface UpdateBannerProps {
  status: UpdateStatus;
  info: UpdateInfo | null;
  progress: number;
  dismissed: boolean;
  isExternalServer?: boolean;
  onInstall: () => void;
  onDismiss: () => void;
}

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

export function UpdateBanner({
  status,
  info,
  progress,
  dismissed,
  isExternalServer = false,
  onInstall,
  onDismiss,
}: UpdateBannerProps) {
  const visible = status === "available";
  const show = visible && !dismissed;

  return (
    <AnimatePresence>
      {show && info && (
        <motion.div
          initial={{ opacity: 0, y: -12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.97 }}
          transition={{ duration: 0.35, ease: EASE_OUT_QUART }}
          className="fixed top-4 right-4 z-[9999] w-[380px]"
        >
          <div className="relative overflow-hidden rounded-xl border border-border/60 bg-background/95 px-5 py-4 shadow-lg backdrop-blur-md">
            {/* Close button */}
            <button
              type="button"
              onClick={onDismiss}
              className="absolute top-3 right-3 flex size-6 items-center justify-center rounded-md text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M11 3L3 11M3 3l8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </button>

            {/* Header */}
            <p className="text-sm font-semibold text-foreground">
              New version: v{info.version}
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {isExternalServer
                ? "Run `unsloth studio update` from your terminal"
                : "A new app update is available to download"}
            </p>

            {/* Actions */}
            <div className="mt-3 flex items-center gap-2">
              <Button size="sm" onClick={onInstall} disabled={isExternalServer}>
                Update Now
              </Button>
              <Button size="sm" variant="outline" disabled>
                Release Notes
              </Button>
              <Button size="sm" variant="ghost" onClick={onDismiss}>
                Later
              </Button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
