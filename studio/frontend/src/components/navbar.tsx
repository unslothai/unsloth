// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
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
  ArrowRight01Icon,
  Book03Icon,
  BubbleChatIcon,
  ChefHatIcon,
  CursorInfo02Icon,
  PackageIcon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useTrainingRuntimeStore } from "@/features/training";
import { usePlatformStore } from "@/config/env";
import { Link, useRouterState } from "@tanstack/react-router";
import { motion } from "motion/react";
import { useState } from "react";
import { TOUR_OPEN_EVENT } from "@/features/tour";

const NAV_ITEMS = [
  { label: "Studio", href: "/studio", icon: ZapIcon, enabled: true },
  { label: "Recipes", href: "/data-recipes", icon: ChefHatIcon, enabled: true },
  { label: "Export", href: "/export", icon: PackageIcon, enabled: true },
  { label: "Chat", href: "/chat", icon: BubbleChatIcon, enabled: true },
];

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

  const chatOnly = usePlatformStore((s) => s.isChatOnly());

  const tourId = getTourId(pathname);

  const openTour = () => {
    if (!tourId) return;
    window.dispatchEvent(
      new CustomEvent(TOUR_OPEN_EVENT, { detail: { id: tourId } }),
    );
  };

  return (
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
          <span className="text-[10px] font-extrabold tracking-[0.12em] text-primary">
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
              chatOnly && item.href !== "/chat";
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

        {/* Right: docs/tour desktop */}
        <div className="hidden items-center justify-self-end gap-2 md:flex">
          <AnimatedThemeToggler
            className="flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground [&_svg]:size-4"
            title="Toggle theme"
            aria-label="Toggle theme"
          />
          <HoverCard openDelay={200} closeDelay={100}>
            <HoverCardTrigger asChild={true}>
              <a
                href="https://unsloth.ai/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-sm font-medium text-emerald-600 hover:text-emerald-700 transition-colors"
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

          <button
            type="button"
            onClick={tourId ? openTour : undefined}
            className={cn(
              "flex h-9 items-center gap-1.5 rounded-md px-3 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
              !tourId && "invisible pointer-events-none",
            )}
            title="Tour"
            aria-hidden={!tourId}
            tabIndex={tourId ? 0 : -1}
          >
            <HugeiconsIcon icon={CursorInfo02Icon} className="size-4" />
            <span className="text-sm font-medium">Tour</span>
          </button>
        </div>

        {/* Right: mobile */}
        <div className="flex items-center gap-2 md:hidden">
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
          <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
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
              <div className="mt-6 flex flex-col gap-2">
                {NAV_ITEMS.filter((item) => item.enabled).map((item) => {
                  const active = pathname === item.href;
                  const disabledByTraining =
                    isTrainingRunning && item.href !== "/studio";
                  const disabledByDevice =
                    chatOnly && item.href !== "/chat";
                  if (disabledByTraining || disabledByDevice) {
                    return (
                      <span
                        key={item.href}
                        className="rounded-md border border-border px-3 py-2 text-sm font-medium text-muted-foreground/40 cursor-not-allowed"
                      >
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
                        "rounded-md border px-3 py-2 text-sm font-medium",
                        active
                          ? "border-foreground bg-foreground text-background"
                          : "border-border text-foreground hover:bg-accent",
                      )}
                    >
                      {item.label}
                    </Link>
                  );
                })}
                <a
                  href="https://unsloth.ai/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-2 rounded-md border border-border px-3 py-2 text-sm font-medium text-foreground hover:bg-accent"
                  onClick={() => setMobileOpen(false)}
                >
                  Learn more (Docs)
                </a>
                {tourId ? (
                  <button
                    type="button"
                    className="rounded-md border border-border px-3 py-2 text-left text-sm font-medium text-foreground hover:bg-accent"
                    onClick={() => {
                      openTour();
                      setMobileOpen(false);
                    }}
                  >
                    Start tour
                  </button>
                ) : null}
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
}
