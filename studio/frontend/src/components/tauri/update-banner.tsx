// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import type {
  DesktopUpdatePolicyMode,
  RetainedUpdateFailure,
  UpdateInfo,
  UpdateStatus,
} from "@/hooks/use-tauri-update";
import type { CopySupportDiagnosticsResult } from "@/lib/tauri-diagnostics";
import { cn } from "@/lib/utils";
import { CircleAlert, Download } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";

interface UpdateBannerProps {
  status: UpdateStatus;
  info: UpdateInfo | null;
  dismissed: boolean;
  lastFailure: RetainedUpdateFailure | null;
  isExternalServer?: boolean;
  updatePolicyMode: DesktopUpdatePolicyMode;
  manualReleaseUrl: string | null;
  // false fills a shared overlay stack; true self-anchors.
  positioned?: boolean;
  onInstall: () => void;
  onDismiss: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

function formatVersion(version: string | null | undefined): string {
  if (!version) return "";
  return version.startsWith("v") ? version : `v${version}`;
}

export function UpdateBanner({
  status,
  info,
  dismissed,
  lastFailure,
  isExternalServer = false,
  updatePolicyMode,
  manualReleaseUrl,
  positioned = true,
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
  const isManualLinuxPackage = updatePolicyMode === "manual_linux_package";
  const installDisabled = isManualLinuxPackage
    ? manualReleaseUrl === null
    : isExternalServer;
  const currentVersion = formatVersion(info?.currentVersion);
  const latestVersion = formatVersion(info?.version);
  const Icon = showFailure ? CircleAlert : Download;

  async function handleCopyDiagnostics() {
    setCopying(true);
    try {
      const result = await onCopyDiagnostics();
      if (result.ok) {
        setManualReport(null);
        setManualMessage(null);
      } else {
        setManualReport(result.report);
        setManualMessage(
          result.error ??
            "Clipboard copy failed. Select and copy the diagnostics below.",
        );
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
          initial={{ opacity: 0, y: 12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.97 }}
          transition={{ duration: 0.35, ease: EASE_OUT_QUART }}
          className={cn(
            positioned
              ? "fixed bottom-4 right-4 z-[9999] w-[calc(100vw-2rem)] max-w-[400px]"
              : "pointer-events-auto w-full",
          )}
          data-testid="tauri-update-banner"
        >
          <div className="relative overflow-hidden rounded-[24px] bg-white px-5 pb-4 pt-5 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-[0_8px_28px_-6px_rgba(0,0,0,0.28)]">
            <button
              type="button"
              onClick={onDismiss}
              className="absolute top-2.5 right-3 flex size-6 items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
              aria-label="Dismiss app update notification"
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

            <div className="flex min-w-0 items-start gap-4 pr-6">
              <Icon
                aria-hidden="true"
                className="mt-1 size-5 shrink-0 text-foreground"
                strokeWidth={1.75}
              />
              <div className="min-w-0">
                <p className="font-heading text-base font-medium text-foreground">
                  {showFailure ? "App update failed" : "New Unsloth version"}
                </p>
                {showFailure ? null : (
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {currentVersion} &rarr;{" "}
                    <span className="font-medium text-foreground">
                      {latestVersion}
                    </span>
                  </p>
                )}
                <p className="mt-1 text-ui-11 text-muted-foreground/70">
                  {showFailure
                    ? "Backend recovered. Diagnostics are still available."
                    : isManualLinuxPackage
                      ? "Open the GitHub release page to install the Linux package"
                      : isExternalServer
                        ? "Run `unsloth studio update` from your terminal"
                        : "A new app update is available"}
                </p>
              </div>
            </div>

            {showFailure && lastFailure && (
              <p className="mt-3 line-clamp-2 text-xs text-destructive">
                {lastFailure.error}
              </p>
            )}

            <div className="mt-4 flex flex-wrap items-center justify-end gap-x-1 gap-y-2">
              {showFailure ? (
                <>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-auto rounded-full px-3 py-2 text-ui-13 font-medium text-foreground"
                    onClick={() => {
                      handleCopyDiagnostics().catch(console.error);
                    }}
                  >
                    {copying ? "Copying..." : "Copy diagnostics"}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-auto rounded-full px-3 py-2 text-ui-13 font-medium text-foreground"
                    onClick={onDismiss}
                  >
                    Later
                  </Button>
                  <Button
                    size="sm"
                    className="-mr-1 h-auto rounded-full px-3.5 py-2 text-ui-13"
                    onClick={onInstall}
                    disabled={installDisabled}
                  >
                    {isManualLinuxPackage ? "Open release page" : "Retry update"}
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-auto rounded-full px-3 py-2 text-ui-13 font-medium text-foreground"
                    onClick={onDismiss}
                  >
                    Remind me later
                  </Button>
                  <Button
                    size="sm"
                    className="-mr-1 h-auto rounded-full px-3.5 py-2 text-ui-13"
                    onClick={onInstall}
                    disabled={installDisabled}
                  >
                    {isManualLinuxPackage ? "Open release page" : "Update"}
                  </Button>
                </>
              )}
            </div>
            {manualMessage && (
              <p className="mt-3 text-xs text-destructive">{manualMessage}</p>
            )}
            {manualReport && (
              <textarea
                readOnly={true}
                value={manualReport}
                onFocus={(event) => event.currentTarget.select()}
                className="mt-2 h-28 w-full resize-none rounded-lg border border-border/50 bg-muted/30 p-2 font-mono text-ui-10 text-muted-foreground"
              />
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
