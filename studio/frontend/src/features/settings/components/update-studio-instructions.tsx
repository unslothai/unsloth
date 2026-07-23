// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { useT } from "@/i18n";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ArrowUpRight01Icon,
  Copy01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import type { ReactElement } from "react";
import { useEffect, useRef, useState } from "react";

const STUDIO_INSTALL_UNIX_CMD =
  "curl -fsSL https://unsloth.ai/install.sh | sh";
const STUDIO_INSTALL_WINDOWS_CMD = "irm https://unsloth.ai/install.ps1 | iex";
const STUDIO_LOCAL_PULL_CMD = "git pull --ff-only";
const STUDIO_LOCAL_INSTALL_UNIX_CMD = "./install.sh --local";
const STUDIO_LOCAL_INSTALL_WINDOWS_CMD = ".\\install.ps1 --local";

const DOCS_INSTALL_URL = "https://unsloth.ai/docs/get-started/install";
const DOCS_UPDATING_URL =
  "https://unsloth.ai/docs/get-started/install/updating";
const DOCS_MAC_URL = "https://unsloth.ai/docs/get-started/install/mac";
const DOCS_WINDOWS_URL =
  "https://unsloth.ai/docs/get-started/install/windows-installation";

export type UpdateShell = "windows" | "unix";
export type UpdateInstallSource =
  | "pypi"
  | "editable"
  | "local_path"
  | "vcs"
  | "local_repo"
  | "unknown";
type UpdateInstallSourceState = UpdateInstallSource | "loading";

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
  const t = useT();
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
        className="min-w-0 flex-1 bg-transparent px-2 py-1.5 font-mono text-ui-11 text-foreground outline-none"
        title={command}
        aria-label={t("settings.about.update.commandText", {
          label: copyLabel,
        })}
      />
      <button
        type="button"
        onClick={handleCopy}
        className="flex shrink-0 items-center justify-center border-l border-border px-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        title={
          copied
            ? t("settings.about.update.copied")
            : t("settings.about.update.copyCommand")
        }
        aria-label={
          copied
            ? t("settings.about.update.commandCopied", { label: copyLabel })
            : t("settings.about.update.copyNamedCommand", {
                label: copyLabel,
              })
        }
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

function DocsLink({
  href,
  label,
}: {
  href: string;
  label: string;
}): ReactElement {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-0.5 font-medium text-muted-foreground hover:text-foreground"
    >
      {label}
      <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
    </a>
  );
}

function UpdateDocsLinks(): ReactElement {
  const t = useT();
  return (
    <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground leading-relaxed">
      {t("settings.about.update.docs")}
      <DocsLink
        href={DOCS_INSTALL_URL}
        label={t("settings.about.update.docsInstall")}
      />
      <DocsLink
        href={DOCS_UPDATING_URL}
        label={t("settings.about.update.docsUpdating")}
      />
      <DocsLink
        href={DOCS_MAC_URL}
        label={t("settings.about.update.docsMac")}
      />
      <DocsLink
        href={DOCS_WINDOWS_URL}
        label={t("settings.about.update.docsWindows")}
      />
    </p>
  );
}

function ShellToggleButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}): ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "inline-flex h-8 items-center justify-center rounded-full px-3.5 text-ui-12 font-medium transition-colors",
        active
          ? "hub-tab-toggle-pill text-foreground"
          : "cursor-pointer text-muted-foreground hover:text-foreground",
      )}
    >
      {label}
    </button>
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
  const t = useT();
  const [shellOverride, setShellOverride] = useState<UpdateShell | null>(null);
  const shell = shellOverride ?? defaultShell;
  const prefersReducedMotion = useReducedMotion();
  const windows = shell === "windows";
  // null means the desktop app: its bundled backend updates through the
  // built-in updater, so terminal commands would target the wrong install.
  const desktopManaged = installSource === null;
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

  if (desktopManaged) {
    return (
      <div className={cn("flex flex-col gap-3", className)}>
        {showTitle ? (
          <p className="shrink-0 whitespace-nowrap text-sm font-semibold font-heading">
            {t("settings.about.update.title")}
          </p>
        ) : null}
        <p className="text-xs text-muted-foreground leading-relaxed">
          {t("settings.about.update.desktopManaged")}
        </p>
        <UpdateDocsLinks />
      </div>
    );
  }

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
            {t("settings.about.update.title")}
          </p>
        ) : null}
        <div className="hub-tab-toggle inline-flex h-8 shrink-0 items-center rounded-full">
          <ShellToggleButton
            active={!windows}
            label="MacOS / Linux"
            onClick={() => setShellOverride("unix")}
          />
          <ShellToggleButton
            active={windows}
            label="Windows"
            onClick={() => setShellOverride("windows")}
          />
        </div>
      </div>
      {loadingInstallSource ? (
        <p className="text-xs text-muted-foreground leading-relaxed">
          {t("settings.about.update.checkingInstall")}
        </p>
      ) : (
        <>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("settings.about.update.installIntro")}
          </p>
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={`install-${shell}`}
              initial={fadeInitial}
              animate={fadeAnimate}
              exit={fadeExit}
              transition={fadeTransition}
            >
              <CopyableCommand
                command={
                  windows ? STUDIO_INSTALL_WINDOWS_CMD : STUDIO_INSTALL_UNIX_CMD
                }
                copyLabel={
                  windows
                    ? t("settings.about.update.installCommandWindows")
                    : t("settings.about.update.installCommandUnix")
                }
              />
            </motion.div>
          </AnimatePresence>
        </>
      )}
      {loadingInstallSource ? null : localInstallSource ? (
        <>
          <p className="text-xs font-semibold text-foreground">
            {t("settings.about.update.localUpdateHeading")}
          </p>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("settings.about.update.localInstallDetected")}
          </p>
          {checkoutInstallSource ? (
            <>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {t("settings.about.update.pullThenUpdate")}
              </p>
              <CopyableCommand
                command={STUDIO_LOCAL_PULL_CMD}
                copyLabel={t("settings.about.update.gitPullCommand")}
              />
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={`local-installer-${shell}`}
                  initial={fadeInitial}
                  animate={fadeAnimate}
                  exit={fadeExit}
                  transition={fadeTransition}
                >
                  <CopyableCommand
                    command={
                      windows
                        ? STUDIO_LOCAL_INSTALL_WINDOWS_CMD
                        : STUDIO_LOCAL_INSTALL_UNIX_CMD
                    }
                    copyLabel={t("settings.about.update.localInstallerCommand")}
                  />
                </motion.div>
              </AnimatePresence>
            </>
          ) : null}
          {packagedSourceInstall ? (
            <>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {t("settings.about.update.sourceInstallDetected")}
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {t("settings.about.update.repoCheckoutFallback")}
              </p>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={`source-installer-${shell}`}
                  initial={fadeInitial}
                  animate={fadeAnimate}
                  exit={fadeExit}
                  transition={fadeTransition}
                >
                  <CopyableCommand
                    command={
                      windows
                        ? STUDIO_LOCAL_INSTALL_WINDOWS_CMD
                        : STUDIO_LOCAL_INSTALL_UNIX_CMD
                    }
                    copyLabel={t("settings.about.update.localInstallerCommand")}
                  />
                </motion.div>
              </AnimatePresence>
            </>
          ) : null}
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("settings.about.update.restartAfterUpdate")}
          </p>
          <UpdateDocsLinks />
        </>
      ) : unknownInstallSource ? (
        <>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("settings.about.update.unknownInstall")}
          </p>
          <p className="text-xs font-semibold text-foreground">
            {t("settings.about.update.localUpdateHeading")}
          </p>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("settings.about.update.localCheckout")}
          </p>
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={`local-installer-${shell}`}
              initial={fadeInitial}
              animate={fadeAnimate}
              exit={fadeExit}
              transition={fadeTransition}
            >
              <CopyableCommand
                command={
                  windows
                    ? STUDIO_LOCAL_INSTALL_WINDOWS_CMD
                    : STUDIO_LOCAL_INSTALL_UNIX_CMD
                }
                copyLabel={t("settings.about.update.localInstallerCommand")}
              />
            </motion.div>
          </AnimatePresence>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("settings.about.update.restartAfterUpdate")}
          </p>
          <UpdateDocsLinks />
        </>
      ) : (
        <>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("settings.about.update.restartAfterUpdate")}
          </p>
          <UpdateDocsLinks />
        </>
      )}
    </div>
  );
}
