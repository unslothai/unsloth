// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import type { ReactElement } from "react";
import { useEffect, useRef, useState } from "react";

const STUDIO_UPDATE_CMD = "unsloth studio update";
const STUDIO_UPDATE_FALLBACK_UNIX_CMD =
  "curl -fsSL https://unsloth.ai/install.sh | sh";
const STUDIO_UPDATE_FALLBACK_WINDOWS_CMD =
  "irm https://unsloth.ai/install.ps1 | iex";

export type UpdateShell = "windows" | "unix";

function getStudioUpdateInstructionLine(shell: UpdateShell): string {
  return shell === "windows" ? "Open PowerShell and run:" : "Open Terminal and run:";
}

function CopyableCommand({
  command,
  copyLabel,
}: {
  command: string;
  copyLabel: string;
}): ReactElement {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const handleCopy = async () => {
    if (!(await copyToClipboard(command))) {
      return;
    }
    setCopied(true);
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    timerRef.current = setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex min-w-0 items-stretch overflow-hidden rounded-md border border-border bg-muted/40">
      <input
        type="text"
        readOnly
        value={command}
        className="min-w-0 flex-1 bg-transparent px-2 py-1.5 font-mono text-[11px] text-foreground outline-none"
        title={command}
        aria-label={`${copyLabel} text`}
      />
      <button
        type="button"
        onClick={handleCopy}
        className="flex shrink-0 items-center justify-center border-l border-border px-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        title={copied ? "Copied" : "Copy command"}
        aria-label={copied ? `${copyLabel} copied` : `Copy ${copyLabel}`}
      >
        {copied ? (
          <HugeiconsIcon icon={Tick02Icon} className="size-4 text-emerald-600" />
        ) : (
          <HugeiconsIcon icon={Copy01Icon} className="size-4" />
        )}
      </button>
    </div>
  );
}

export function UpdateStudioInstructions({
  className,
  defaultShell,
  showTitle = true,
}: {
  className?: string;
  defaultShell: UpdateShell;
  showTitle?: boolean;
}): ReactElement {
  const [shell, setShell] = useState<UpdateShell>(defaultShell);
  const prefersReducedMotion = useReducedMotion();
  const windows = shell === "windows";
  const fadeTransition = prefersReducedMotion
    ? { duration: 0 }
    : { duration: 0.16, ease: [0.165, 0.84, 0.44, 1] as const };
  const fadeInitial = prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 2 };
  const fadeAnimate = { opacity: 1, y: 0 };
  const fadeExit = prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: -2 };

  useEffect(() => {
    setShell(defaultShell);
  }, [defaultShell]);

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      <div
        className={cn(
          "flex items-center gap-3",
          showTitle ? "justify-between" : "justify-start",
        )}
      >
        {showTitle ? (
          <p className="shrink-0 whitespace-nowrap text-sm font-semibold font-heading">
            Update Unsloth Studio
          </p>
        ) : null}
        <div className="flex shrink-0 items-center gap-0.5 text-[11px]">
          <button
            type="button"
            onClick={() => setShell("windows")}
            className={cn(
              "px-0.5 py-0.5 font-medium transition-colors",
              windows
                ? "text-foreground"
                : "text-muted-foreground hover:text-emerald-600",
            )}
            aria-pressed={windows}
          >
            Windows
          </button>
          <span className="text-border">/</span>
          <button
            type="button"
            onClick={() => setShell("unix")}
            className={cn(
              "px-0.5 py-0.5 font-medium transition-colors",
              !windows
                ? "text-foreground"
                : "text-muted-foreground hover:text-emerald-600",
            )}
            aria-pressed={!windows}
          >
            macOS/Linux
          </button>
        </div>
      </div>
      <AnimatePresence mode="wait" initial={false}>
        <motion.p
          key={`instruction-${shell}`}
          initial={fadeInitial}
          animate={fadeAnimate}
          exit={fadeExit}
          transition={fadeTransition}
          className="text-xs text-muted-foreground leading-relaxed"
        >
          {getStudioUpdateInstructionLine(shell)}
        </motion.p>
      </AnimatePresence>
      <CopyableCommand command={STUDIO_UPDATE_CMD} copyLabel="update command" />
      <p className="text-xs text-muted-foreground leading-relaxed">
        If that fails or unsloth studio update is unavailable, run:
      </p>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={`fallback-${shell}`}
          initial={fadeInitial}
          animate={fadeAnimate}
          exit={fadeExit}
          transition={fadeTransition}
        >
          <CopyableCommand
            command={
              windows
                ? STUDIO_UPDATE_FALLBACK_WINDOWS_CMD
                : STUDIO_UPDATE_FALLBACK_UNIX_CMD
            }
            copyLabel="fallback command"
          />
        </motion.div>
      </AnimatePresence>
      <p className="text-xs text-muted-foreground leading-relaxed">
        Restart Studio after updating for changes to take effect.
      </p>
    </div>
  );
}
