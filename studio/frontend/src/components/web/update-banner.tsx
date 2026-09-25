// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { ReleaseNotesPanel } from "@/components/update/release-notes-panel";
import { type DeviceType, usePlatformStore } from "@/config/env";
import { useWebUpdateCheck } from "@/hooks/use-web-update-check";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { Download } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { type ReactElement, useEffect, useRef, useState } from "react";

// macOS, Linux and WSL update via the POSIX installer; only native Windows (PowerShell) needs the
// irm one-liner. Any non-windows device_type (incl. wsl) resolves to the curl command below.
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
  const [notesVersion, setNotesVersion] = useState<string | null>(null);
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
  // Keyed by version so a new offer collapses the panel.
  const notesOpen = status != null && notesVersion === status.latestVersion;

  return (
    <AnimatePresence>
      {status ? (
        <motion.div
          initial={{ opacity: 0, y: 12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.97 }}
          transition={{ duration: 0.35, ease: EASE_OUT_QUART }}
          className={cn(
            // Wider than the other overlays: notes preview plus three buttons.
            positioned
              ? "fixed bottom-4 right-4 z-[9999] w-[calc(100vw-2rem)] max-w-[calc(448px*var(--ui-space-scale,1))]"
              : cn(
                  "pointer-events-auto flex w-[calc(100vw-2rem)] max-w-[calc(448px*var(--ui-space-scale,1))] shrink-0 flex-col",
                  // Only rendered notes may shrink in the capped rail. Without
                  // them, shrink-0 keeps the compact card at its natural height.
                  // How far it may shrink is the surface's own content floor
                  // below, not a written-out number: a constant calibrated at
                  // one type size stops covering the card the moment anything
                  // inside it scales, and the buttons are what gets cut.
                  "has-[[data-slot=update-release-notes]]:shrink",
                ),
          )}
          data-testid="web-update-banner"
        >
          {/* Paint the full floor even when the notes content is short.
              No min-h-0 and no overflow-hidden here on purpose: both of them
              set this box's automatic minimum size to zero, which is what made
              the card need a hand-written floor at all. Left alone it floors
              itself at header + notes + actions, and the notes panel carries
              its own min-h-0 and clipping, so it stays the one part that
              yields. That is the floor the constant was approximating, except
              it is measured rather than guessed and it holds at every type
              size. */}
          <div className="relative flex max-h-[calc(100dvh_-_2rem)] grow flex-col rounded-[24px] bg-white px-5 pb-4 pt-5 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-[0_8px_28px_-6px_var(--background)]">

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

            <div className="flex min-w-0 shrink-0 items-start gap-4 pr-6">
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

            <ReleaseNotesPanel
              version={status.latestVersion}
              open={notesOpen}
              releaseNotesUrl={RELEASE_NOTES_URL}
            />

            {/* one row at one type size; wraps only on narrow viewports, and
                never compresses on a short one */}
            <div className="mt-4 flex shrink-0 flex-wrap items-center justify-between gap-y-2">
              <Button
                size="sm"
                variant="ghost"
                className="-ml-2 h-auto whitespace-nowrap rounded-full px-2.5 py-2 text-ui-13 font-medium text-foreground"
                onClick={() =>
                  setNotesVersion(notesOpen ? null : status.latestVersion)
                }
                aria-expanded={notesOpen}
                data-testid="web-update-release-notes-toggle"
              >
                {notesOpen ? "Hide release notes" : "Show release notes"}
              </Button>
              {/* wrap + right-align so buttons stack instead of clipping on very narrow banners */}
              <div className="flex flex-wrap items-center justify-end gap-x-1 gap-y-2">
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-auto whitespace-nowrap rounded-full px-2.5 py-2 text-ui-13 font-medium text-foreground"
                  onClick={snooze}
                  data-testid="web-update-snooze-button"
                >
                  Remind me later
                </Button>
                <Button
                  size="sm"
                  // -mr optically aligns the filled pill's edge with the card padding
                  className="-mr-1 h-auto whitespace-nowrap rounded-full px-3 py-2 text-ui-13"
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
