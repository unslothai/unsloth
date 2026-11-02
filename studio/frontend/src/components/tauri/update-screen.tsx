// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseBackendDownloadProgress } from "@/components/tauri/backend-download-progress";
import { DiagnosticsCopyActions } from "@/components/tauri/diagnostics-copy-actions";
import { LogDetails } from "@/components/tauri/log-details";
import { shouldUseNativeMacWindowTitlebar } from "@/components/tauri/window-titlebar";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import type { UpdateStatus } from "@/hooks/use-tauri-update";
import { useT } from "@/i18n";
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
        className="h-[calc(60px*var(--ui-space-scale,1))] w-[calc(60px*var(--ui-space-scale,1))] object-contain"
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

type Translate = ReturnType<typeof useT>;

function statusLabel(status: UpdateStatus, t: Translate): string {
  switch (status) {
    case "updating-backend":
      return t("desktopUpdate.updatingBackend");
    case "downloading":
      return t("desktopUpdate.downloadingApp");
    case "installing":
      return t("desktopUpdate.installing");
    case "error":
      return t("desktopUpdate.failed");
    default:
      return t("desktopUpdate.updating");
  }
}

function statusSubtext(status: UpdateStatus, progress: number, t: Translate): string {
  switch (status) {
    case "updating-backend":
      return t("desktopUpdate.updatingBackendDetail");
    case "downloading":
      return t("desktopUpdate.percentDownloaded", { percent: progress });
    case "installing":
      return t("desktopUpdate.restartShortly");
    case "error":
      return t("desktopUpdate.failedDetail");
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
  const t = useT();
  const isError = status === "error";
  const backendDownload =
    status === "updating-backend"
      ? parseBackendDownloadProgress(logs.at(-1) ?? "")
      : null;
  const downloadProgress =
    status === "downloading" ? progress : backendDownload?.percent;

  return (
    <div className="box-border flex h-full w-full flex-col items-center overflow-y-auto bg-background pb-6 pt-[var(--studio-startup-top-inset,0px)]">
      {shouldUseNativeMacWindowTitlebar() && (
        <div
          data-tauri-drag-region={true}
          aria-hidden="true"
          className="pointer-events-auto fixed inset-x-0 top-0 z-50 h-[var(--studio-mac-titlebar-height,34px)] select-none"
        />
      )}
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
              {statusLabel(status, t)}
            </p>
            <p className="text-sm text-muted-foreground">
              {statusSubtext(status, progress, t)}
            </p>

            {backendDownload && (
              <div className="mt-2 w-full max-w-sm text-xs text-muted-foreground">
                <p className="break-words">
                  {t("desktopUpdate.downloadingFile", { file: backendDownload.file })}
                </p>
                <p>{backendDownload.detail}</p>
              </div>
            )}

            {downloadProgress != null && (
              <div className="mt-2 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-muted">
                <progress
                  aria-label={
                    backendDownload
                      ? t("desktopUpdate.downloadingFile", { file: backendDownload.file })
                      : t("desktopUpdate.downloadingAppLabel")
                  }
                  max={100}
                  value={downloadProgress}
                  className="sr-only"
                />
                <motion.div
                  className="h-full rounded-full bg-primary"
                  initial={{ width: 0 }}
                  animate={{ width: `${downloadProgress}%` }}
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
                <Button variant="dark" size="hero" onClick={onRetry}>
                  {t("desktopStartup.retry")}
                </Button>
                <Button variant="muted" size="hero" onClick={onSkipRestart}>
                  {t("desktopUpdate.skipRestart")}
                </Button>
              </DiagnosticsCopyActions>
            )}

            <LogDetails
              showLabel={t("desktopStartup.details.showUpdate")}
              hideLabel={t("desktopStartup.details.hideUpdate")}
              lines={logs}
            />
          </div>
        </motion.div>
      </div>
    </div>
  );
}
