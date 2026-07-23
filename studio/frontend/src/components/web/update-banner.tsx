// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { type DeviceType, usePlatformStore } from "@/config/env";
import { useWebUpdateCheck } from "@/hooks/use-web-update-check";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { Download } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { type ReactElement, useEffect, useRef, useState } from "react";

// macOS, Linux and WSL update via the POSIX installer; only native Windows
// (PowerShell) needs the irm one-liner. Any non-windows device_type (incl. wsl)
// resolves to the curl command below.
const STUDIO_INSTALL_UNIX_CMD = "curl -fsSL https://unsloth.ai/install.sh | sh";
const STUDIO_INSTALL_WINDOWS_CMD = "irm https://unsloth.ai/install.ps1 | iex";
const RELEASE_NOTES_URL = "https://unsloth.ai/docs/new/changelog";
const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

function installCommandForDevice(deviceType: DeviceType): string {
  return deviceType === "windows"
    ? STUDIO_INSTALL_WINDOWS_CMD
    : STUDIO_INSTALL_UNIX_CMD;
}

interface WebUpdateBannerProps {
  enabled?: boolean;
  // false: fill the parent instead of self-anchoring, so it can stack with the
  // llama.cpp banner. true (default) keeps standalone mounts working.
  positioned?: boolean;
}

export function WebUpdateBanner({
  enabled = true,
  positioned = true,
}: WebUpdateBannerProps): ReactElement | null {
  const { status, dismiss, snooze } = useWebUpdateCheck({ enabled });
  const deviceType = usePlatformStore((s) => s.deviceType);
  const installCmd = installCommandForDevice(deviceType);
  const [copiedVersion, setCopiedVersion] = useState<string | null>(null);
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (dismissTimerRef.current) {
        clearTimeout(dismissTimerRef.current);
      }
    };
  }, []);

  if (isTauri) {
    return null;
  }

  async function handleCopyCommand() {
    if (!(await copyToClipboard(installCmd))) {
      return;
    }
    setCopiedVersion(status?.latestVersion ?? null);
    if (dismissTimerRef.current) {
      clearTimeout(dismissTimerRef.current);
    }
    // Copying is not updating: snooze instead of dismissing, so the banner
    // returns on the next launch if the install is still behind.
    dismissTimerRef.current = setTimeout(() => snooze(), 1200);
  }

  const copied = status != null && copiedVersion === status.latestVersion;

  return (
    <AnimatePresence>
      {status ? (
        <motion.div
          initial={{ opacity: 0, y: 12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.97 }}
          transition={{ duration: 0.35, ease: EASE_OUT_QUART }}
          className={cn(
            positioned
              ? "fixed bottom-4 right-4 z-[9999] w-[calc(100vw-32px)] max-w-[400px]"
              : "pointer-events-auto w-full",
          )}
          data-testid="web-update-banner"
        >
          <div className="relative overflow-hidden rounded-[24px] bg-white px-5 pb-4 pt-5 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-[0_8px_28px_-6px_rgba(0,0,0,0.28)]">
            <button
              type="button"
              onClick={dismiss}
              className="absolute top-2.5 right-3 flex size-6 items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
              aria-label="Dismiss update notification"
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
              <Download
                aria-hidden="true"
                className="mt-1 size-5 shrink-0 text-foreground"
                strokeWidth={1.75}
              />
              <div className="min-w-0">
                <p className="font-heading text-base font-medium text-foreground">
                  New Unsloth version
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {status.currentVersion} &rarr;{" "}
                  <span className="font-medium text-foreground">
                    {status.latestVersion}
                  </span>
                </p>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap items-center justify-between gap-y-2">
              <a
                href={RELEASE_NOTES_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="-ml-2 whitespace-nowrap rounded-full px-2.5 py-2 text-[0.8125rem] font-medium text-foreground transition-colors hover:bg-muted"
                data-testid="web-update-release-notes-link"
              >
                Release notes
              </a>
              {/* wrap + right-align so buttons stack instead of clipping on very narrow banners */}
              <div className="flex flex-wrap items-center justify-end gap-x-1 gap-y-2">
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-auto rounded-full px-3 py-2 text-[0.8125rem] font-medium text-foreground"
                  onClick={snooze}
                  data-testid="web-update-snooze-button"
                >
                  Remind me later
                </Button>
                <Button
                  size="sm"
                  // -mr optically aligns the filled pill's edge with the card padding
                  className="-mr-1 h-auto rounded-full px-3.5 py-2 text-[0.8125rem]"
                  onClick={handleCopyCommand}
                  data-testid="web-update-copy-button"
                >
                  {copied ? "Copied" : "Copy command"}
                </Button>
              </div>
            </div>
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
