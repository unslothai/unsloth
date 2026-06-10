// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useWebUpdateCheck } from "@/hooks/use-web-update-check";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { AnimatePresence, motion } from "motion/react";
import { type ReactElement, useEffect, useRef, useState } from "react";

const STUDIO_UPDATE_CMD = "unsloth studio update";
const RELEASE_NOTES_URL = "https://unsloth.ai/docs/new/changelog";
const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

interface WebUpdateBannerProps {
  enabled?: boolean;
}

export function WebUpdateBanner({
  enabled = true,
}: WebUpdateBannerProps): ReactElement | null {
  const { status, dismiss } = useWebUpdateCheck({ enabled });
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
    if (!(await copyToClipboard(STUDIO_UPDATE_CMD))) {
      return;
    }
    setCopiedVersion(status?.latestVersion ?? null);
    if (dismissTimerRef.current) {
      clearTimeout(dismissTimerRef.current);
    }
    dismissTimerRef.current = setTimeout(() => dismiss(), 900);
  }

  return (
    <AnimatePresence>
      {status ? (
        <motion.div
          initial={{ opacity: 0, y: -12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.97 }}
          transition={{ duration: 0.35, ease: EASE_OUT_QUART }}
          className="fixed top-4 right-4 z-[9999] w-[calc(100vw-2rem)] max-w-[380px]"
        >
          <div className="corner-squircle relative overflow-hidden border border-border/60 bg-background/95 px-5 py-4 shadow-lg backdrop-blur-md">
            <button
              type="button"
              onClick={dismiss}
              className="absolute top-3 right-3 flex size-6 items-center justify-center rounded-md text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
              aria-label="Dismiss update notification"
            >
              <svg
                aria-hidden="true"
                width="14"
                height="14"
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

            <div className="flex items-start gap-2 pr-5">
              <span className="text-lg" aria-hidden="true">
                🦥
              </span>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-foreground">
                  Package update available: {status.latestVersion}
                </p>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                  Installed package: {status.currentVersion}. To update Studio,
                  run this in your terminal, then restart Studio.
                </p>
              </div>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                className="corner-squircle"
                onClick={handleCopyCommand}
              >
                {copiedVersion === status.latestVersion
                  ? "Copied"
                  : "Copy command"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="corner-squircle"
                asChild={true}
              >
                <a
                  href={RELEASE_NOTES_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Release notes
                </a>
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="corner-squircle"
                onClick={dismiss}
              >
                Later
              </Button>
            </div>
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
