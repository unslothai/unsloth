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
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { TOUR_OPEN_EVENT, isTourListening } from "@/features/tour";
import { useT } from "@/i18n";
import {
  ChevronLeftStandardIcon,
  ChevronRightStandardIcon,
} from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
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

function whenReady(attempt: () => boolean, timeoutMs = 3000): void {
  const deadline = Date.now() + timeoutMs;
  const tick = () => {
    if (attempt() || Date.now() > deadline) return;
    window.setTimeout(tick, 120);
  };
  window.setTimeout(tick, 120);
}

function ProgressDots({
  pages,
  page,
  onPick,
}: {
  pages: OnboardingPageId[];
  page: OnboardingPageId;
  onPick: (index: number) => void;
}) {
  const t = useT();
  const current = pages.indexOf(page);
  return (
    <div className="pointer-events-none absolute inset-x-0 flex justify-center">
      {pages.map((p, i) => (
        <button
          key={p}
          type="button"
          aria-label={t("onboarding.progress", {
            current: i + 1,
            total: pages.length,
          })}
          aria-current={i === current ? "step" : undefined}
          onClick={() => onPick(i)}
          className="group pointer-events-auto flex h-6 items-center rounded-full px-[3px] outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <span
            className={cn(
              "block h-1.5 rounded-full transition-all duration-300",
              i === current
                ? "bg-control-accent w-5"
                : i < current
                  ? "bg-muted-foreground group-hover:bg-foreground w-1.5"
                  : "bg-muted-foreground/30 group-hover:bg-muted-foreground w-1.5",
            )}
          />
        </button>
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

  const goTo = useCallback(
    (target: number) => {
      if (target === index) return;
      if (target > index) record(page);
      setDirection(target > index ? 1 : -1);
      setIndex(target);
    },
    [index, page, record],
  );

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
    <Dialog open={true} onOpenChange={(open) => !open && skipAll()}>
      <DialogContent
        ref={cardRef}
        data-onboarding=""
        showCloseButton={false}
        // Connect opens Settings over the guide; clicks and Esc there must not dismiss it.
        onInteractOutside={(e) => {
          if (useSettingsDialogStore.getState().open) e.preventDefault();
        }}
        onEscapeKeyDown={(e) => {
          if (useSettingsDialogStore.getState().open) e.preventDefault();
        }}
        onOpenAutoFocus={(e) => {
          e.preventDefault();
          primaryRef.current?.focus();
        }}
        className="flex max-h-[calc(100dvh-var(--studio-window-chrome-top,0px)-2rem)] min-h-[min(calc(41.5rem*var(--ui-space-scale,1)),calc(100dvh-var(--studio-window-chrome-top,0px)-2rem))] flex-col gap-0 overflow-hidden p-0 sm:max-w-[calc(40rem*var(--ui-space-scale,1))] sm:rounded-[2rem]"
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
        <div className="absolute top-4 right-4 z-10">
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
        <div className="relative flex min-h-0 flex-1 flex-col overflow-x-hidden overflow-y-auto px-6 py-12 [mask-image:linear-gradient(to_bottom,transparent,black_calc(2.5rem*var(--ui-space-scale,1)),black_calc(100%-2.5rem*var(--ui-space-scale,1)),transparent)] sm:px-14">
          <AnimatePresence
            mode="wait"
            initial={false}
            custom={direction}
            onExitComplete={keepFocusInside}
          >
            <motion.div
              key={page}
              className="mx-auto my-auto flex w-full max-w-[calc(30rem*var(--ui-space-scale,1))] flex-col"
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
        <div className="relative flex shrink-0 items-center justify-between gap-3 px-6 pt-1 pb-6 sm:px-8 sm:pb-6">
          <ProgressDots pages={pages} page={page} onPick={goTo} />
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
                icon={ChevronLeftStandardIcon}
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
              icon={ChevronRightStandardIcon}
              className="size-4"
              strokeWidth={2}
            />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
