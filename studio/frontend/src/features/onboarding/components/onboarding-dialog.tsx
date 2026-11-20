// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { usePlatformStore } from "@/config/env";
import { useUserProfileStore } from "@/features/profile/stores/user-profile-store";
import { triggerShortcut } from "@/features/settings/hooks/use-shortcut";
import { TOUR_OPEN_EVENT, isTourListening } from "@/features/tour";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useNavigate } from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useRef, useState } from "react";
import { saveOnboarding } from "../api";
import {
  ONBOARDING_PAGES,
  type OnboardingPageId,
  START_ROUTES,
  START_TOURS,
  type StartChoice,
  isNumberedPage,
} from "../lib/pages";
import { useOnboardingStore } from "../store";
import { useDeviceSummary } from "../use-device-summary";
import {
  DeviceStep,
  DoneStep,
  LookStep,
  ProfileStep,
  StartStep,
  WelcomeStep,
} from "./onboarding-steps";

const PLACE_LABELS: Record<StartChoice, TranslationKey> = {
  chat: "onboarding.start.chatTitle",
  train: "onboarding.start.trainTitle",
  recipes: "onboarding.start.recipesTitle",
  hub: "onboarding.start.hubTitle",
};

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

function ProgressDots({
  pages,
  page,
}: { pages: OnboardingPageId[]; page: OnboardingPageId }) {
  const numbered = pages.filter(isNumberedPage);
  const current = numbered.indexOf(page);

  return (
    <div
      aria-hidden="true"
      className={cn(
        "flex h-1.5 items-center justify-center gap-1.5 transition-opacity duration-300",
        current < 0 && "opacity-0",
      )}
    >
      {numbered.map((p, i) => (
        <span
          key={p}
          className={cn(
            "h-1.5 rounded-full transition-all duration-300",
            i === current
              ? "w-5 bg-control-accent"
              : i < current
                ? "w-1.5 bg-muted-foreground"
                : "w-1.5 bg-muted-foreground/30",
          )}
        />
      ))}
    </div>
  );
}

const PILL =
  "h-10 min-w-30 rounded-full px-5 [font-family:var(--font-heading)] text-[0.875rem] font-semibold tracking-[0.02em]";

export function OnboardingDialog({ pages }: { pages: OnboardingPageId[] }) {
  const t = useT();
  const navigate = useNavigate();
  const close = useOnboardingStore((s) => s.close);
  const trainAvailable = usePlatformStore((s) => !s.isTrainUnavailable());
  const device = useDeviceSummary();
  const [index, setIndex] = useState(0);
  const [direction, setDirection] = useState(1);
  const [start, setStart] = useState<StartChoice>("chat");
  const [tour, setTour] = useState(false);
  const seen = useRef(new Set<OnboardingPageId>());
  const contentRef = useRef<HTMLDivElement>(null);
  const primaryRef = useRef<HTMLButtonElement>(null);
  const page = pages[index];
  const last = index === pages.length - 1;
  const numbered = pages.filter(isNumberedPage);
  // A Train pick made before detection settled falls back to Chat once the verdict says no.
  const choice = start === "train" && !trainAvailable ? "chat" : start;
  const place = t(PLACE_LABELS[choice]);
  const hasName = useUserProfileStore((s) => s.displayName.trim() !== "");

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
    void navigate({ to: START_ROUTES[choice] });
    if (tour) {
      const id = START_TOURS[choice];
      whenReady(() => {
        if (!isTourListening(id)) return false;
        window.dispatchEvent(
          new CustomEvent(TOUR_OPEN_EVENT, { detail: { id } }),
        );
        return true;
      });
    } else if (choice === "chat") {
      whenReady(() => triggerShortcut("openModelPicker"));
    }
  }, [choice, close, navigate, page, pages, record, tour]);

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

  const step =
    page === "welcome" ? (
      <WelcomeStep />
    ) : page === "profile" ? (
      <ProfileStep onSubmit={next} />
    ) : page === "look" ? (
      <LookStep />
    ) : page === "device" ? (
      <DeviceStep summary={device} />
    ) : page === "start" ? (
      <StartStep
        value={choice}
        onChange={setStart}
        trainAvailable={trainAvailable}
      />
    ) : (
      <DoneStep
        start={choice}
        place={place}
        tour={tour}
        onTourChange={setTour}
      />
    );

  const primaryLabel =
    page === "welcome"
      ? t("onboarding.getStarted")
      : last
        ? t("onboarding.done.open", { place })
        : page === "profile" && !hasName
          ? t("onboarding.skip")
          : t("onboarding.continue");

  const current = numbered.indexOf(page);

  // A step swap unmounts whatever held focus; left on the body, keystrokes reach the composer behind.
  const keepFocusInside = useCallback(() => {
    if (!contentRef.current?.contains(document.activeElement)) {
      primaryRef.current?.focus();
    }
  }, []);

  return (
    <Dialog open={true} onOpenChange={(open) => !open && skipAll()}>
      <DialogContent
        showCloseButton={false}
        overlayTone="dense"
        onInteractOutside={(e) => e.preventDefault()}
        ref={contentRef}
        onOpenAutoFocus={(e) => {
          e.preventDefault();
          primaryRef.current?.focus();
        }}
        className="flex h-[min(calc(37rem*var(--ui-space-scale,1)),calc(100dvh-var(--studio-window-chrome-top,0px)-2rem))] flex-col gap-0 overflow-hidden p-0 sm:max-w-152"
      >
        <DialogTitle className="sr-only">
          {t("onboarding.dialogTitle")}
        </DialogTitle>
        <DialogDescription className="sr-only">
          {current >= 0
            ? t("onboarding.progress", {
                current: current + 1,
                total: numbered.length,
              })
            : t("onboarding.welcome.body")}
        </DialogDescription>
        <div className="flex h-12 shrink-0 items-end justify-center">
          <ProgressDots pages={pages} page={page} />
        </div>
        <div className="relative min-h-0 flex-1 overflow-x-hidden overflow-y-auto px-7 pt-6 pb-4 sm:px-10">
          <AnimatePresence
            mode="wait"
            initial={false}
            custom={direction}
            onExitComplete={keepFocusInside}
          >
            <motion.div
              key={page}
              className="flex min-h-full flex-col"
              custom={direction}
              variants={{
                enter: (d: number) => ({
                  opacity: 0,
                  y: 12 * d,
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
        <div className="flex h-18 shrink-0 items-center justify-between gap-3 px-6 sm:px-8">
          {page === "welcome" ? (
            <Button
              variant="ghost"
              className={cn(PILL, "min-w-0 px-4 text-muted-foreground")}
              onClick={skipAll}
            >
              {t("onboarding.skipSetup")}
            </Button>
          ) : index > 0 ? (
            <Button
              variant="ghost"
              className={cn(PILL, "min-w-0 px-4")}
              onClick={back}
            >
              {t("onboarding.back")}
            </Button>
          ) : (
            <span />
          )}
          <Button
            ref={primaryRef}
            variant="primary"
            className={cn(
              PILL,
              "bg-control-accent text-control-accent-foreground hover:bg-control-accent/85",
            )}
            onClick={next}
          >
            {primaryLabel}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
