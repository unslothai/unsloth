// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
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
const STUDIO_LOCAL_PULL_CMD = "git pull --ff-only";
const STUDIO_LOCAL_UPDATE_CMD = "unsloth studio update --local";
const STUDIO_LOCAL_FALLBACK_UNIX_CMD = "./install.sh --local";
const STUDIO_LOCAL_FALLBACK_WINDOWS_CMD = ".\\install.ps1 --local";

export type UpdateShell = "windows" | "unix";
export type UpdateInstallSource =
  | "pypi"
  | "editable"
  | "local_path"
  | "vcs"
  | "local_repo"
  | "unknown";
type UpdateInstallSourceState = UpdateInstallSource | "loading";

function getStudioUpdateInstructionLine(shell: UpdateShell): string {
  return shell === "windows"
    ? "Open PowerShell and run:"
    : "Open Terminal and run:";
}

function isLocalInstallSource(
  installSource?: UpdateInstallSourceState | null,
): boolean {
  return Boolean(
    installSource &&
      installSource !== "pypi" &&
      installSource !== "unknown" &&
      installSource !== "loading",
  );
}

function isUnknownInstallSource(
  installSource?: UpdateInstallSourceState | null,
): boolean {
  return installSource === "unknown";
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
        readOnly={true}
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
          <HugeiconsIcon
            icon={Tick02Icon}
            className="size-4 text-emerald-600"
          />
        ) : (
          <HugeiconsIcon icon={Copy01Icon} className="size-4" />
        )}
      </button>
    </div>
  );
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: keep source-specific update guidance in one component so the command matrix stays visible.
export function UpdateStudioInstructions({
  className,
  defaultShell,
  installSource,
  showTitle = true,
}: {
  className?: string;
  defaultShell: UpdateShell;
  installSource?: UpdateInstallSourceState | null;
  showTitle?: boolean;
}): ReactElement {
  const [shell, setShell] = useState<UpdateShell>(defaultShell);
  const prefersReducedMotion = useReducedMotion();
  const windows = shell === "windows";
  const localInstallSource = isLocalInstallSource(installSource);
  const checkoutInstallSource =
    installSource === "editable" || installSource === "local_repo";
  const packagedSourceInstall =
    installSource === "vcs" || installSource === "local_path";
  const loadingInstallSource = installSource === "loading";
  const unknownInstallSource = isUnknownInstallSource(installSource);
  const fadeTransition = prefersReducedMotion
    ? { duration: 0 }
    : { duration: 0.16, ease: [0.165, 0.84, 0.44, 1] as const };
  const fadeInitial = prefersReducedMotion
    ? { opacity: 1 }
    : { opacity: 0, y: 2 };
  const fadeAnimate = { opacity: 1, y: 0 };
  const fadeExit = prefersReducedMotion
    ? { opacity: 1 }
    : { opacity: 0, y: -2 };

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
              windows
                ? "text-muted-foreground hover:text-emerald-600"
                : "text-foreground",
            )}
            aria-pressed={!windows}
          >
            macOS/Linux
          </button>
        </div>
      </div>
      {loadingInstallSource ? (
        <p className="text-xs text-muted-foreground leading-relaxed">
          Checking how Studio was installed…
        </p>
      ) : localInstallSource ? (
        <>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Source or local install detected. To avoid replacing it with PyPI,
            update from the checkout or source you originally installed from.
          </p>
          {checkoutInstallSource ? (
            <>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Pull latest changes from your Unsloth repo checkout, then update
                Studio locally:
              </p>
              <CopyableCommand
                command={STUDIO_LOCAL_PULL_CMD}
                copyLabel="git pull command"
              />
              <CopyableCommand
                command={STUDIO_LOCAL_UPDATE_CMD}
                copyLabel="local update command"
              />
              <p className="text-xs text-muted-foreground leading-relaxed">
                If the Studio update command is unavailable, run the local
                installer from that checkout:
              </p>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={`local-fallback-${shell}`}
                  initial={fadeInitial}
                  animate={fadeAnimate}
                  exit={fadeExit}
                  transition={fadeTransition}
                >
                  <CopyableCommand
                    command={
                      windows
                        ? STUDIO_LOCAL_FALLBACK_WINDOWS_CMD
                        : STUDIO_LOCAL_FALLBACK_UNIX_CMD
                    }
                    copyLabel="local installer command"
                  />
                </motion.div>
              </AnimatePresence>
            </>
          ) : null}
          {packagedSourceInstall ? (
            <>
              <p className="text-xs text-muted-foreground leading-relaxed">
                This looks like a source or VCS package install. Reinstall from
                the original local path or Git URL you used.
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                If you still have the Unsloth repo checkout, run the local
                installer from that checkout:
              </p>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={`source-fallback-${shell}`}
                  initial={fadeInitial}
                  animate={fadeAnimate}
                  exit={fadeExit}
                  transition={fadeTransition}
                >
                  <CopyableCommand
                    command={
                      windows
                        ? STUDIO_LOCAL_FALLBACK_WINDOWS_CMD
                        : STUDIO_LOCAL_FALLBACK_UNIX_CMD
                    }
                    copyLabel="local installer command"
                  />
                </motion.div>
              </AnimatePresence>
            </>
          ) : null}
          <p className="text-xs text-muted-foreground leading-relaxed">
            Restart Studio after updating for changes to take effect.
          </p>
        </>
      ) : unknownInstallSource ? (
        <>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Studio could not detect how it was installed. Check how you
            installed Studio first, then choose the matching update path.
          </p>
          <p className="text-xs text-muted-foreground leading-relaxed">
            For curl or PyPI installs, run:
          </p>
          <CopyableCommand
            command={STUDIO_UPDATE_CMD}
            copyLabel="update command"
          />
          <p className="text-xs text-muted-foreground leading-relaxed">
            For local checkout installs, update from that checkout instead and
            use the local update command:
          </p>
          <CopyableCommand
            command={STUDIO_LOCAL_UPDATE_CMD}
            copyLabel="local update command"
          />
          <p className="text-xs text-muted-foreground leading-relaxed">
            Restart Studio after updating for changes to take effect.
          </p>
        </>
      ) : (
        <>
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
          <CopyableCommand
            command={STUDIO_UPDATE_CMD}
            copyLabel="update command"
          />
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
        </>
      )}
    </div>
  );
}
