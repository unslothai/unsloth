// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { AnimatedThemeToggler } from "@/components/ui/animated-theme-toggler";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import {
  ArrowReloadHorizontalIcon,
  ArrowRight01Icon,
  Cancel01Icon,
  Book03Icon,
  BubbleChatIcon,
  ChefHatIcon,
  Copy01Icon,
  CursorInfo02Icon,
  PackageIcon,
  Tick02Icon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { useTrainingRuntimeStore } from "@/features/training";
import { usePlatformStore } from "@/config/env";
import { Link, useRouterState } from "@tanstack/react-router";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import type { ReactElement } from "react";
import { useEffect, useRef, useState } from "react";
import { TOUR_OPEN_EVENT } from "@/features/tour";
import { ShutdownDialog } from "@/components/shutdown-dialog";

const NAV_ITEMS = [
  { label: "Studio", href: "/studio", icon: ZapIcon, enabled: true },
  { label: "Recipes", href: "/data-recipes", icon: ChefHatIcon, enabled: true },
  { label: "Export", href: "/export", icon: PackageIcon, enabled: true },
  { label: "Chat", href: "/chat", icon: BubbleChatIcon, enabled: true },
];

const STUDIO_UPDATE_CMD = "unsloth studio update";
const STUDIO_UPDATE_FALLBACK_UNIX_CMD =
  "curl -fsSL https://unsloth.ai/install.sh | sh";
const STUDIO_UPDATE_FALLBACK_WINDOWS_CMD =
  "irm https://unsloth.ai/install.ps1 | iex";

type UpdateShell = "windows" | "unix";

function getDefaultUpdateShell(deviceType: string): UpdateShell {
  return deviceType === "windows" ? "windows" : "unix";
}

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

  const handleCopy = () => {
    if (!copyToClipboard(command)) {
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

function UpdateStudioInstructions({
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

function getTourId(pathname: string): "studio" | "chat" | "export" | null {
  if (pathname === "/studio") return "studio";
  if (pathname === "/chat") return "chat";
  if (pathname === "/export") return "export";
  return null;
}

export function Navbar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isTrainingRunning = useTrainingRuntimeStore((s) => s.isTrainingRunning);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [mobileUpdateOpen, setMobileUpdateOpen] = useState(false);
  const [shutdownOpen, setShutdownOpen] = useState(false);

  const deviceType = usePlatformStore((s) => s.deviceType);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const defaultUpdateShell = getDefaultUpdateShell(deviceType);

  // Warn before closing the tab only when training is running (data loss risk).
  // We store the handler in a ref so removeUnloadHandler() can clean it up
  // before the "Server stopped" page renders.
  const unloadHandlerRef = useRef<((e: BeforeUnloadEvent) => void) | null>(null);

  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (!useTrainingRuntimeStore.getState().isTrainingRunning) return;
      e.preventDefault();
      e.returnValue = "";
    };
    unloadHandlerRef.current = handler;
    window.addEventListener("beforeunload", handler);
    return () => {
      window.removeEventListener("beforeunload", handler);
    };
  }, []);

  const removeUnloadHandler = () => {
    if (unloadHandlerRef.current) {
      window.removeEventListener("beforeunload", unloadHandlerRef.current);
      unloadHandlerRef.current = null;
    }
  };

  const tourId = getTourId(pathname);

  const openTour = () => {
    if (!tourId) return;
    window.dispatchEvent(
      new CustomEvent(TOUR_OPEN_EVENT, { detail: { id: tourId } }),
    );
  };

  return (
    <>
    <header className="relative top-0 z-40 h-16 w-full">
      <div className="mx-auto grid h-full max-w-7xl grid-cols-[1fr_auto_1fr] items-center px-4 sm:px-6">
        {/* Left: logo */}
        <Link to={chatOnly ? "/chat" : "/studio"} className="flex items-center gap-1.5 justify-self-start select-none">
          <img
            src="/blacklogo.png"
            alt="Unsloth"
            className="h-9 w-auto dark:hidden"
          />
          <img
            src="/whitelogo.png"
            alt="Unsloth"
            className="hidden h-9 w-auto dark:block"
          />
          <span className="relative -top-[1px] inline-flex items-center text-[10px] font-extrabold leading-none tracking-[0.12em] text-primary">
            BETA
          </span>
        </Link>

        {/* Center: pill nav */}
        <nav
          data-tour="navbar"
          className="hidden items-center rounded-full border border-border bg-card p-1 ring-1 ring-foreground/5 md:flex"
        >
          {NAV_ITEMS.map((item) => {
            const active =
              pathname === item.href || pathname.startsWith(`${item.href}/`);
            const disabledByTraining =
              isTrainingRunning && item.href !== "/studio";
            const disabledByDevice =
              chatOnly && item.href !== "/chat" && item.href !== "/data-recipes";
            if (!item.enabled || disabledByTraining || disabledByDevice) {
              return (
                <span
                  key={item.href}
                  className="relative rounded-full px-3 py-1.5 text-sm font-medium text-muted-foreground/40 cursor-not-allowed"
                >
                  {item.label}
                </span>
              );
            }
            return (
              <Link
                key={item.href}
                to={item.href}
                className={cn(
                  "relative rounded-full px-3 py-1.5 text-sm font-medium transition-colors",
                  active
                    ? "text-background"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {active && (
                  <motion.span
                    layoutId="nav-pill"
                    className="absolute inset-0 rounded-full bg-foreground"
                    transition={{
                      type: "spring",
                      stiffness: 500,
                      damping: 35,
                      mass: 0.5,
                    }}
                  />
                )}
                <span className="relative z-10 flex items-center">
                  <motion.span
                    initial={false}
                    animate={{
                      width: active ? 14 : 0,
                      marginLeft: active ? -4 : 0,
                      marginRight: active ? 4 : 0,
                      opacity: active ? 1 : 0,
                    }}
                    transition={{ duration: 0.2, ease: [0.165, 0.84, 0.44, 1] }}
                    className="inline-flex shrink-0 items-center justify-center overflow-hidden"
                  >
                    <HugeiconsIcon
                      icon={item.icon}
                      className="size-3.5 -mt-px shrink-0"
                    />
                  </motion.span>
                  {item.label}
                </span>
              </Link>
            );
          })}
        </nav>

        {/* Right: docs/tour desktop — one wrapper per control so flex gap is even (HoverCard roots can confuse flex spacing). */}
        <div className="hidden items-center justify-self-end gap-0 md:flex">
          <div className="flex shrink-0 items-center">
            <AnimatedThemeToggler
              className="flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground [&_svg]:size-4"
              title="Toggle theme"
              aria-label="Toggle theme"
            />
          </div>
          <div className="flex shrink-0 items-center">
            <HoverCard openDelay={200} closeDelay={100}>
              <HoverCardTrigger asChild={true}>
                <a
                  href="https://unsloth.ai/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex h-9 items-center gap-1.5 rounded-md px-3 text-sm font-medium text-emerald-600 transition-colors hover:bg-accent hover:text-emerald-700 dark:hover:text-emerald-400"
                >
                  <HugeiconsIcon icon={Book03Icon} className="size-4" />
                  Learn more
                </a>
              </HoverCardTrigger>
              <HoverCardContent align="end" className="w-80 p-0">
                <a
                  href="https://unsloth.ai/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group/card flex flex-col gap-1 p-4 no-underline"
                >
                  <p className="text-sm font-semibold font-heading">
                    Unsloth Documentation
                  </p>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Guides on fine-tuning LLMs 2x faster with 70% less memory.
                    Covers LoRA, QLoRA, data formatting, and deployment.
                  </p>
                  <span className="mt-1 flex items-center gap-1 text-xs font-medium text-emerald-600 group-hover/card:underline">
                    Visit docs
                    <HugeiconsIcon icon={ArrowRight01Icon} className="size-3" />
                  </span>
                </a>
              </HoverCardContent>
            </HoverCard>
          </div>
          {tourId ? (
            <div className="flex shrink-0 items-center">
              <button
                type="button"
                onClick={openTour}
                className="flex h-9 items-center gap-1.5 rounded-md px-3 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                title="Tour"
              >
                <HugeiconsIcon icon={CursorInfo02Icon} className="size-4" />
                <span className="text-sm font-medium">Tour</span>
              </button>
            </div>
          ) : null}
          <div className="flex shrink-0 items-center">
            <HoverCard openDelay={200} closeDelay={100}>
              <HoverCardTrigger asChild={true}>
                <button
                  type="button"
                  className="flex h-9 items-center gap-1.5 rounded-md px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                  aria-label="How to update Unsloth Studio"
                >
                  <HugeiconsIcon icon={ArrowReloadHorizontalIcon} className="size-4" />
                  Update
                </button>
              </HoverCardTrigger>
              <HoverCardContent align="end" className="w-[22.5rem] p-0">
                <UpdateStudioInstructions
                  className="p-4"
                  defaultShell={defaultUpdateShell}
                />
              </HoverCardContent>
            </HoverCard>
          </div>
          <div className="flex shrink-0 items-center">
            <button
              type="button"
              onClick={() => setShutdownOpen(true)}
              className="flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              title="Shut down Unsloth Studio server"
              aria-label="Shut down Unsloth Studio server"
            >
              <HugeiconsIcon icon={Cancel01Icon} className="size-5" />
            </button>
          </div>
        </div>

        {/* Right: mobile */}
        <div className="col-start-3 flex items-center gap-2 justify-self-end md:hidden">
          {tourId ? (
            <button
              type="button"
              onClick={openTour}
              className="flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              title="Tour"
            >
              <HugeiconsIcon icon={CursorInfo02Icon} className="size-4" />
            </button>
          ) : null}
          <Sheet
            open={mobileOpen}
            onOpenChange={(open) => {
              setMobileOpen(open);
              if (!open) setMobileUpdateOpen(false);
            }}
          >
            <SheetTrigger asChild={true}>
              <button
                type="button"
                className="rounded-md border border-border px-3 py-1.5 text-sm font-medium text-foreground"
                aria-label="Open navigation menu"
              >
                Menu
              </button>
            </SheetTrigger>
            <SheetContent side="right" className="w-[300px] p-4">
              <SheetHeader>
                <SheetTitle>Navigate</SheetTitle>
              </SheetHeader>
              <div className="mt-6 flex max-h-[calc(100dvh-8rem)] flex-col gap-2 overflow-y-auto pr-1">
                {NAV_ITEMS.filter((item) => item.enabled).map((item) => {
                  const active = pathname === item.href;
                  const disabledByTraining =
                    isTrainingRunning && item.href !== "/studio";
                  const disabledByDevice =
                    chatOnly && item.href !== "/chat" && item.href !== "/data-recipes";
                  if (disabledByTraining || disabledByDevice) {
                    return (
                      <span
                        key={item.href}
                        className="flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm font-medium text-muted-foreground/40 cursor-not-allowed"
                      >
                        <HugeiconsIcon icon={item.icon} className="size-4" />
                        {item.label}
                      </span>
                    );
                  }
                  return (
                    <Link
                      key={item.href}
                      to={item.href}
                      onClick={() => setMobileOpen(false)}
                      className={cn(
                        "flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium",
                        active
                          ? "border-foreground bg-foreground text-background"
                          : "border-border text-foreground hover:bg-accent",
                      )}
                    >
                      <HugeiconsIcon icon={item.icon} className="size-4" />
                      {item.label}
                    </Link>
                  );
                })}
                <a
                  href="https://unsloth.ai/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-3 flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm font-medium text-foreground hover:bg-accent"
                  onClick={() => setMobileOpen(false)}
                >
                  <HugeiconsIcon icon={Book03Icon} className="size-4" />
                  Learn more (Docs)
                </a>
                {tourId ? (
                  <button
                    type="button"
                    className="flex items-center gap-2 rounded-md border border-border px-3 py-2 text-left text-sm font-medium text-foreground hover:bg-accent"
                    onClick={() => {
                      openTour();
                      setMobileOpen(false);
                    }}
                  >
                    <HugeiconsIcon icon={CursorInfo02Icon} className="size-4" />
                    Start tour
                  </button>
                ) : null}
                <Collapsible
                  open={mobileUpdateOpen}
                  onOpenChange={setMobileUpdateOpen}
                  className="rounded-md border border-border"
                >
                  <CollapsibleTrigger asChild={true}>
                    <button
                      type="button"
                      className="flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm font-medium text-foreground transition-colors hover:bg-accent"
                      aria-label="Toggle update instructions"
                    >
                      <span className="flex items-center gap-2">
                        <HugeiconsIcon icon={ArrowReloadHorizontalIcon} className="size-4" />
                        Update Unsloth Studio
                      </span>
                      <HugeiconsIcon
                        icon={ArrowRight01Icon}
                        className={cn(
                          "size-4 text-muted-foreground transition-transform",
                          mobileUpdateOpen && "rotate-90",
                        )}
                      />
                    </button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="border-t border-border p-3 pt-2">
                    <UpdateStudioInstructions
                      defaultShell={defaultUpdateShell}
                      showTitle={false}
                    />
                  </CollapsibleContent>
                </Collapsible>
                <button
                  type="button"
                  className="mt-3 flex items-center gap-2 rounded-md border border-border px-3 py-2 text-left text-sm font-medium text-foreground hover:bg-accent"
                  onClick={() => {
                    setMobileOpen(false);
                    setShutdownOpen(true);
                  }}
                >
                  <HugeiconsIcon icon={Cancel01Icon} className="size-5" />
                  Quit Unsloth Studio
                </button>
                <div className="mt-2 flex items-center justify-between rounded-md border border-border px-3 py-2">
                  <span className="text-sm font-medium text-foreground">Theme</span>
                  <AnimatedThemeToggler
                    className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground [&_svg]:size-4"
                    title="Toggle theme"
                    aria-label="Toggle theme"
                  />
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>

    <ShutdownDialog
      open={shutdownOpen}
      onOpenChange={setShutdownOpen}
      onBeforeShutdown={removeUnloadHandler}
    />
    </>
  );
}
