// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import type { RetainedUpdateFailure, UpdateInfo, UpdateStatus } from "@/hooks/use-tauri-update";
import type { CopySupportDiagnosticsResult } from "@/lib/tauri-diagnostics";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";

interface UpdateBannerProps {
  status: UpdateStatus;
  info: UpdateInfo | null;
  dismissed: boolean;
  lastFailure: RetainedUpdateFailure | null;
  isExternalServer?: boolean;
  onInstall: () => void;
  onDismiss: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

export function UpdateBanner({
  status,
  info,
  dismissed,
  lastFailure,
  isExternalServer = false,
  onInstall,
  onDismiss,
  onCopyDiagnostics,
}: UpdateBannerProps) {
  const [copying, setCopying] = useState(false);
  const [manualReport, setManualReport] = useState<string | null>(null);
  const [manualMessage, setManualMessage] = useState<string | null>(null);
  const showFailure = Boolean(lastFailure) && !dismissed;
  const showAvailable = status === "available" && !dismissed && !showFailure;
  const show = showFailure || (showAvailable && Boolean(info));

  async function handleCopyDiagnostics() {
    setCopying(true);
    try {
      const result = await onCopyDiagnostics();
      if (result.ok) {
        setManualReport(null);
        setManualMessage(null);
      } else {
        setManualReport(result.report);
        setManualMessage(result.error ?? "Clipboard copy failed. Select and copy the diagnostics below.");
      }
    } catch (error) {
      setManualReport(null);
      setManualMessage(`Diagnostics copy failed: ${String(error)}`);
    } finally {
      setCopying(false);
    }
  }

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, y: -12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.97 }}
          transition={{ duration: 0.35, ease: EASE_OUT_QUART }}
          className="fixed top-4 right-4 z-[9999] w-[380px]"
        >
          <div className="corner-squircle relative overflow-hidden border border-border/60 bg-background/95 px-5 py-4 shadow-lg backdrop-blur-md">
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
            <div className="flex items-center gap-2">
              <span className="text-lg">🦥</span>
              <div>
                <p className="text-sm font-semibold text-foreground">
                  {showFailure ? "App update failed" : `New version: v${info?.version}`}
                </p>
                <p className="text-xs text-muted-foreground">
                  {showFailure
                    ? "Backend recovered. Diagnostics are still available."
                    : isExternalServer
                      ? "Run `unsloth studio update` from your terminal"
                      : "A new app update is available"}
                </p>
              </div>
            </div>

            {/* Retained failure */}
            {showFailure && lastFailure && (
              <p className="mt-3 line-clamp-2 text-xs text-destructive">
                {lastFailure.error}
              </p>
            )}

            {/* Actions */}
            <div className="mt-3 flex items-center gap-2">
              {showFailure ? (
                <>
                  <Button size="sm" variant="outline" className="corner-squircle" onClick={() => void handleCopyDiagnostics()}>
                    {copying ? "Copying..." : "Copy Diagnostics"}
                  </Button>
                  <Button size="sm" className="corner-squircle" onClick={onInstall} disabled={isExternalServer}>
                    Retry Update
                  </Button>
                </>
              ) : (
                <>
                  <Button size="sm" className="corner-squircle" onClick={onInstall} disabled={isExternalServer}>
                    Update Now
                  </Button>
                  <Button size="sm" variant="outline" className="corner-squircle" disabled>
                    Release Notes
                  </Button>
                </>
              )}
              <Button size="sm" variant="ghost" className="corner-squircle" onClick={onDismiss}>
                Later
              </Button>
            </div>
            {manualMessage && (
              <p className="mt-3 text-xs text-destructive">{manualMessage}</p>
            )}
            {manualReport && (
              <textarea
                readOnly
                value={manualReport}
                onFocus={(event) => event.currentTarget.select()}
                className="mt-2 h-28 w-full resize-none rounded-lg border border-border/50 bg-muted/30 p-2 font-mono text-[10px] text-muted-foreground"
              />
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
