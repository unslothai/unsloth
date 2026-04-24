// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { UpdateStatus } from "@/hooks/use-tauri-update";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef } from "react";

interface UpdateScreenProps {
  status: UpdateStatus;
  logs: string[];
  progress: number;
  error: string | null;
  onRetry: () => void;
  onSkipRestart: () => void;
}

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

function Spinner({ size = 24 }: { size?: number }) {
  return (
    <span
      className="inline-block animate-spin rounded-full border-2 border-primary border-t-transparent"
      style={{ width: size, height: size, animationDuration: "0.8s" }}
    />
  );
}

function Logo() {
  return (
    <div className="flex flex-col items-center gap-4">
      <img src="/sticker.png" alt="Unsloth" className="h-[72px] w-[72px] object-contain" />
      <img src="/studio.png" alt="Unsloth Studio" className="h-auto w-[250px] object-contain dark:invert" />
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

function LogViewer({ logs }: { logs: string[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  if (logs.length === 0) return null;

  return (
    <div
      ref={scrollRef}
      className="mt-4 h-[180px] w-full max-w-xl overflow-y-auto rounded-lg border border-border/40 bg-muted/30 p-3 font-mono text-[11px] leading-relaxed text-muted-foreground"
    >
      {logs.map((line, i) => (
        <div key={i} className="whitespace-pre-wrap break-all">
          {line}
        </div>
      ))}
    </div>
  );
}

export function UpdateScreen({
  status,
  logs,
  progress,
  error,
  onRetry,
  onSkipRestart,
}: UpdateScreenProps) {
  const isError = status === "error";

  return (
    <div className="flex h-screen w-full items-center justify-center bg-background">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: EASE_OUT_QUART }}
        className="flex w-full max-w-xl flex-col items-center px-6"
      >
        <Logo />

        <div className="mt-8 flex flex-col items-center gap-2">
          {!isError && <Spinner />}
          <p className="text-sm font-semibold text-foreground">
            {statusLabel(status)}
          </p>
          <p className="text-xs text-muted-foreground">
            {statusSubtext(status, progress)}
          </p>
        </div>

        {/* Download progress bar */}
        {status === "downloading" && (
          <div className="mt-4 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-muted">
            <motion.div
              className="h-full rounded-full bg-primary"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        )}

        {/* Error display */}
        <AnimatePresence>
          {isError && error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 w-full max-w-xl rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3"
            >
              <p className="text-xs text-destructive">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error actions */}
        {isError && (
          <div className="mt-4 flex items-center gap-2">
            <button
              type="button"
              className="rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/80"
              onClick={onRetry}
            >
              Retry
            </button>
            <button
              type="button"
              className="rounded-lg bg-muted px-5 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-muted/80"
              onClick={onSkipRestart}
            >
              Skip & Restart
            </button>
          </div>
        )}

        {/* Log viewer */}
        <LogViewer logs={logs} />
      </motion.div>
    </div>
  );
}
