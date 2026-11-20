// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { triggerShortcut } from "@/features/settings/hooks/use-shortcut";
import { TOUR_OPEN_EVENT, isTourListening } from "@/features/tour";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Cancel01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useRef, useState } from "react";
import { saveOnboarding } from "../api";
import {
  CHAT_TOUR,
  ONBOARDING_PAGES,
  type OnboardingPageId,
} from "../lib/pages";
import { useOnboardingStore } from "../store";
import { useDeviceSummary } from "../use-device-summary";
import { LookStep } from "./look-steps";
import { DoneStep, WelcomeStep } from "./onboarding-steps";

const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;

// The destination mounts after navigation, so its tour and shortcut answer a few frames later.
function whenReady(attempt: () => boolean, timeoutMs = 3000): void {
  const deadline = Date.now() + timeoutMs;
  const tick = () => {
    if (attempt() || Date.now() > deadline) return;
    window.setTimeout(tick, 120);
  };
  window.setTimeout(tick, 120);
}

/** Drawn once by the frame, so the bar holds still while the steps change under it. */
function ProgressBar({
  pages,
  page,
}: { pages: OnboardingPageId[]; page: OnboardingPageId }) {
  const current = pages.indexOf(page);
  return (
    <div aria-hidden="true" className="flex flex-1 items-center gap-1.5">
      {pages.map((p, i) => (
        <span
          key={p}
          className="bg-muted-foreground/20 h-1 flex-1 overflow-hidden rounded-full"
        >
          <motion.span
            className="bg-control-accent block h-full rounded-full"
            initial={false}
            animate={{ width: i <= current ? "100%" : "0%" }}
            transition={{ duration: 0.45, ease: EASE_OUT_EXPO }}
          />
        </span>
      ))}
    </div>
  );
}

const PILL =
  "h-11 rounded-full text-sm font-semibold [font-family:var(--font-heading)] tracking-[0.01em]";

export function OnboardingDialog({ pages }: { pages: OnboardingPageId[] }) {
  const t = useT();
  const navigate = useNavigate();
  const close = useOnboardingStore((s) => s.close);
  const device = useDeviceSummary();
  const [index, setIndex] = useState(0);
  const [direction, setDirection] = useState(1);
  const [tour, setTour] = useState(false);
  const seen = useRef(new Set<OnboardingPageId>());
  const cardRef = useRef<HTMLDivElement>(null);
  const primaryRef = useRef<HTMLButtonElement>(null);
  const page = pages[index];
  const last = index === pages.length - 1;

  const record = useCallback((id: OnboardingPageId, completed = false) => {
    seen.current.add(id);
    // Best effort: a failed save shows the remaining pages again next time, nothing worse.
    void saveOnboarding([...seen.current], completed).catch(() => {});
  }, []);

  const skipAll = useCallback(() => {
    void saveOnboarding([...ONBOARDING_PAGES], true).catch(() => {});
    close();
  }, [close]);

  const finish = useCallback(() => {
    for (const p of pages) seen.current.add(p);
    record(page, true);
    close();
    void navigate({ to: "/chat" });
    if (tour) {
      whenReady(() => {
        if (!isTourListening(CHAT_TOUR)) return false;
        window.dispatchEvent(
          new CustomEvent(TOUR_OPEN_EVENT, { detail: { id: CHAT_TOUR } }),
        );
        return true;
      });
    } else {
      whenReady(() => triggerShortcut("openModelPicker"));
    }
  }, [close, navigate, page, pages, record, tour]);

  const next = useCallback(() => {
    if (last) {
      finish();
      return;
    }
    record(page);
    setDirection(1);
    setIndex((i) => i + 1);
  }, [finish, last, page, record]);

  const back = useCallback(() => {
    setDirection(-1);
    setIndex((i) => Math.max(0, i - 1));
  }, []);

  // A step swap unmounts whatever held focus; left on the body, keystrokes reach the composer behind.
  const keepFocusInside = useCallback(() => {
    if (!cardRef.current?.contains(document.activeElement)) {
      primaryRef.current?.focus();
    }
  }, []);

  const step =
    page === "welcome" ? (
      <WelcomeStep onSubmit={next} />
    ) : page === "look" ? (
      <LookStep />
    ) : (
      <DoneStep device={device} tour={tour} onTourChange={setTour} />
    );

  const primaryLabel =
    page === "welcome"
      ? t("onboarding.getStarted")
      : last
        ? t("onboarding.done.open")
        : t("onboarding.continue");

  const current = pages.indexOf(page);

  return (
    // Clicking outside or Esc closes it as Skip setup, so Chat is always one click away.
    <Dialog open={true} onOpenChange={(open) => !open && skipAll()}>
      <DialogContent
        ref={cardRef}
        data-onboarding=""
        showCloseButton={false}
        onOpenAutoFocus={(e) => {
          e.preventDefault();
          primaryRef.current?.focus();
        }}
        className="flex max-h-[calc(100dvh-var(--studio-window-chrome-top,0px)-2rem)] flex-col gap-0 overflow-hidden p-0 sm:max-w-[calc(40rem*var(--ui-space-scale,1))] sm:rounded-[2rem]"
      >
        <DialogTitle className="sr-only">
          {t("onboarding.dialogTitle")}
        </DialogTitle>
        <DialogDescription className="sr-only">
          {t("onboarding.progress", {
            current: current + 1,
            total: pages.length,
          })}
        </DialogDescription>
        <div className="relative flex shrink-0 items-center gap-4 pt-6 pr-5 pl-8 sm:pl-10">
          <ProgressBar pages={pages} page={page} />
          <DialogClose asChild={true}>
            <Button
              variant="ghost"
              size="icon-sm"
              className="text-muted-foreground rounded-full"
              aria-label={t("onboarding.close")}
            >
              <HugeiconsIcon
                icon={Cancel01Icon}
                className="size-4"
                strokeWidth={2}
              />
            </Button>
          </DialogClose>
        </div>
        <div className="relative min-h-0 flex-1 overflow-x-hidden overflow-y-auto px-6 pt-10 pb-10 sm:px-14">
          <AnimatePresence
            mode="wait"
            initial={false}
            custom={direction}
            onExitComplete={keepFocusInside}
          >
            <motion.div
              key={page}
              className="mx-auto flex min-h-full w-full max-w-[calc(30rem*var(--ui-space-scale,1))] flex-col"
              custom={direction}
              variants={{
                enter: (d: number) => ({
                  opacity: 0,
                  y: 14 * d,
                  filter: "blur(8px)",
                }),
                center: { opacity: 1, y: 0, filter: "blur(0px)" },
                exit: (d: number) => ({
                  opacity: 0,
                  y: -8 * d,
                  filter: "blur(4px)",
                  transition: { duration: 0.16, ease: "easeIn" },
                }),
              }}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{
                duration: 0.42,
                ease: EASE_OUT_EXPO,
                opacity: { duration: 0.3 },
              }}
            >
              {step}
            </motion.div>
          </AnimatePresence>
        </div>
        <div className="relative flex shrink-0 items-center justify-between gap-3 px-6 pb-6 sm:px-10 sm:pb-8">
          {page === "welcome" ? (
            <button
              type="button"
              className="text-muted-foreground hover:text-foreground px-2 text-sm font-medium transition-colors"
              onClick={skipAll}
            >
              {t("onboarding.skipSetup")}
            </button>
          ) : index > 0 ? (
            <Button
              variant="muted"
              size="icon-lg"
              className="size-11 rounded-full"
              aria-label={t("onboarding.back")}
              onClick={back}
            >
              <HugeiconsIcon
                icon={ArrowLeft01Icon}
                className="size-4.5"
                strokeWidth={2}
              />
            </Button>
          ) : (
            <span />
          )}
          <Button
            ref={primaryRef}
            variant="primary"
            className={cn(
              PILL,
              "bg-control-accent text-control-accent-foreground hover:bg-control-accent/85 min-w-36 gap-2 pr-5 pl-6",
            )}
            onClick={next}
          >
            {primaryLabel}
            <HugeiconsIcon
              icon={ArrowRight01Icon}
              className="size-4"
              strokeWidth={2}
            />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
