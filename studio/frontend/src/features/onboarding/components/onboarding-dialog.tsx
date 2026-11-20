// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { usePlatformStore } from "@/config/env";
import { ChatBackground } from "@/features/chat-background";
import { useUserProfileStore } from "@/features/profile/stores/user-profile-store";
import { triggerShortcut } from "@/features/settings/hooks/use-shortcut";
import { TOUR_OPEN_EVENT, isTourListening } from "@/features/tour";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { ArrowLeft01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { Dialog as DialogPrimitive } from "radix-ui";
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
import { LookStep, SceneStep } from "./look-steps";
import {
  DeviceStep,
  DoneStep,
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

/** Drawn once by the frame, so the dots hold still while the steps change under them. */
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
              ? "w-6 bg-control-accent"
              : i < current
                ? "w-1.5 bg-muted-foreground/70"
                : "w-1.5 bg-muted-foreground/25",
          )}
        />
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
  const trainAvailable = usePlatformStore((s) => !s.isTrainUnavailable());
  const device = useDeviceSummary();
  const [index, setIndex] = useState(0);
  const [direction, setDirection] = useState(1);
  const [start, setStart] = useState<StartChoice>("chat");
  const [tour, setTour] = useState(false);
  const seen = useRef(new Set<OnboardingPageId>());
  const cardRef = useRef<HTMLDivElement>(null);
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

  // A step swap unmounts whatever held focus; left on the body, keystrokes reach the composer behind.
  const keepFocusInside = useCallback(() => {
    if (!cardRef.current?.contains(document.activeElement)) {
      primaryRef.current?.focus();
    }
  }, []);

  const step =
    page === "welcome" ? (
      <WelcomeStep />
    ) : page === "profile" ? (
      <ProfileStep onSubmit={next} />
    ) : page === "look" ? (
      <LookStep />
    ) : page === "scene" ? (
      <SceneStep />
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

  return (
    <DialogPrimitive.Root
      open={true}
      onOpenChange={(open) => !open && skipAll()}
    >
      <DialogPrimitive.Portal>
        <DialogPrimitive.Content
          data-auth-wallpaper=""
          data-onboarding=""
          onOpenAutoFocus={(e) => {
            e.preventDefault();
            primaryRef.current?.focus();
          }}
          className="bg-background data-open:animate-in data-open:fade-in-0 fixed inset-x-0 top-[var(--studio-window-chrome-top,0px)] bottom-0 z-50 flex items-center justify-center overflow-hidden duration-300 outline-none sm:p-8"
        >
          <ChatBackground backdrop={true} />
          <Card
            ref={cardRef}
            className="relative z-10 flex h-full w-full gap-0 rounded-none py-0 ring-0 sm:h-[min(calc(42rem*var(--ui-space-scale,1)),100%)] sm:max-w-[calc(40rem*var(--ui-space-scale,1))] sm:rounded-[2rem] sm:ring-1 sm:shadow-[0_24px_64px_-24px_rgba(0,0,0,0.35)] dark:sm:shadow-[var(--composer-shadow)]"
          >
            <DialogPrimitive.Title className="sr-only">
              {t("onboarding.dialogTitle")}
            </DialogPrimitive.Title>
            <DialogPrimitive.Description className="sr-only">
              {current >= 0
                ? t("onboarding.progress", {
                    current: current + 1,
                    total: numbered.length,
                  })
                : t("onboarding.welcome.body")}
            </DialogPrimitive.Description>
            <div className="flex h-16 shrink-0 items-end justify-center">
              <ProgressDots pages={pages} page={page} />
            </div>
            <div className="relative min-h-0 flex-1 overflow-x-hidden overflow-y-auto px-6 pt-9 pb-6 sm:px-14">
              <AnimatePresence
                mode="wait"
                initial={false}
                custom={direction}
                onExitComplete={keepFocusInside}
              >
                <motion.div
                  key={page}
                  className="mx-auto flex min-h-full w-full max-w-[calc(28rem*var(--ui-space-scale,1))] flex-col"
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
            <div className="flex shrink-0 items-center justify-between gap-3 px-6 pt-3 pb-6 sm:px-10 sm:pb-9">
              {page === "welcome" ? (
                <Button
                  variant="ghost"
                  className={cn(PILL, "text-muted-foreground px-5")}
                  onClick={skipAll}
                >
                  {t("onboarding.skipSetup")}
                </Button>
              ) : index > 0 ? (
                <Button
                  variant="ghost"
                  className={cn(PILL, "gap-2 pr-5 pl-4")}
                  onClick={back}
                >
                  <HugeiconsIcon
                    icon={ArrowLeft01Icon}
                    className="size-4"
                    strokeWidth={2}
                  />
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
          </Card>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
