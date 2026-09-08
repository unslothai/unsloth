// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DiagnosticsCopyActions } from "@/components/tauri/diagnostics-copy-actions";
import { LogDetails } from "@/components/tauri/log-details";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import type { UpdateStatus } from "@/hooks/use-tauri-update";
import type { CopySupportDiagnosticsResult } from "@/lib/tauri-diagnostics";

import { AnimatePresence, motion } from "motion/react";

interface UpdateScreenProps {
  status: UpdateStatus;
  logs: string[];
  progress: number;
  error: string | null;
  onRetry: () => void;
  onSkipRestart: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

function Logo() {
  return (
    <div className="flex items-center justify-center gap-3">
      <img
        src="/sticker.png"
        alt=""
        aria-hidden="true"
        className="h-[60px] w-[60px] object-contain"
      />
      <span
        className="text-ui-50 font-semibold leading-none tracking-[-0.02em] text-foreground"
        style={{ fontFamily: '"Hellix", sans-serif' }}
      >
        unsloth
      </span>
    </div>
  );
}

function statusLabel(status: UpdateStatus): string {
  switch (status) {
    case "updating-backend":
      return "Updating backend...";
    case "downloading":
      return "Downloading app update...";
    case "installing":
      return "Installing update...";
    case "error":
      return "Update failed";
    default:
      return "Updating...";
  }
}

function statusSubtext(status: UpdateStatus, progress: number): string {
  switch (status) {
    case "updating-backend":
      return "This may take a few minutes. Do not close the app.";
    case "downloading":
      return `${progress}% downloaded`;
    case "installing":
      return "The app will restart shortly.";
    case "error":
      return "Something went wrong during the update.";
    default:
      return "";
  }
}

export function UpdateScreen({
  status,
  logs,
  progress,
  error,
  onRetry,
  onSkipRestart,
  onCopyDiagnostics,
}: UpdateScreenProps) {
  const isError = status === "error";

  return (
    <div className="box-border flex h-full w-full flex-col items-center overflow-y-auto bg-background pb-6 pt-[var(--studio-startup-top-inset,0px)]">
      <div className="flex min-h-0 w-full max-w-md flex-1 items-center justify-center px-6">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.2, ease: EASE_OUT_QUART }}
          className="flex h-full w-full flex-col items-center text-center"
        >
          <div className="flex flex-1 items-center">
            <Logo />
          </div>

          <div className="mb-10 flex w-full flex-col items-center gap-2">
            {!isError && <Spinner className="size-6 text-primary" />}
            <p
              className={
                isError
                  ? "text-sm font-medium text-destructive"
                  : "text-sm font-bold text-foreground"
              }
            >
              {statusLabel(status)}
            </p>
            <p className="text-sm text-muted-foreground">
              {statusSubtext(status, progress)}
            </p>

            {status === "downloading" && (
              <div className="mt-2 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-muted">
                <motion.div
                  className="h-full rounded-full bg-primary"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            )}

            <AnimatePresence>
              {isError && error && (
                <motion.p
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="max-w-md text-xs text-muted-foreground"
                >
                  {error}
                </motion.p>
              )}
            </AnimatePresence>

            {isError && (
              <DiagnosticsCopyActions onCopyDiagnostics={onCopyDiagnostics}>
                <Button size="hero" onClick={onRetry}>
                  Retry
                </Button>
                <Button variant="muted" size="hero" onClick={onSkipRestart}>
                  Skip & Restart
                </Button>
              </DiagnosticsCopyActions>
            )}

            <LogDetails label="update details" lines={logs} />
          </div>
        </motion.div>
      </div>
    </div>
  );
}
