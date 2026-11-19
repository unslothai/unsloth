// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useComposerModelPickerSource } from "@/components/assistant-ui/composer-model-picker/context";
import { MascotImg, publicAssetUrl } from "@/components/mascot-img";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowRight01Icon,
  Brain03Icon,
  Cancel01Icon,
  CloudIcon,
  Compass01Icon,
  Download01Icon,
  GridViewIcon,
  Search01Icon,
  SparklesIcon,
  StarIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type FC, useRef, useState, useSyncExternalStore } from "react";

// Literal copies live in Reset preferences (general-tab.tsx) and ACCOUNT_CHROME_KEYS.
export const MODEL_PICKER_ANNOUNCEMENT_DISMISSED_KEY =
  "unsloth_model_picker_intro_dismissed";

const listeners = new Set<() => void>();

function readDismissed(): boolean {
  try {
    return (
      localStorage.getItem(MODEL_PICKER_ANNOUNCEMENT_DISMISSED_KEY) === "true"
    );
  } catch {
    return true;
  }
}

function dismiss(): void {
  try {
    localStorage.setItem(MODEL_PICKER_ANNOUNCEMENT_DISMISSED_KEY, "true");
  } catch {
    // storage unavailable
  }
  for (const listener of listeners) {
    listener();
  }
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (
      event.key === MODEL_PICKER_ANNOUNCEMENT_DISMISSED_KEY ||
      event.key === null
    ) {
      listener();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

const FEATURES = [
  { icon: GridViewIcon, label: "Every model in one place" },
  { icon: Download01Icon, label: "Download in one pick" },
  { icon: Brain03Icon, label: "Reasoning built in" },
  { icon: StarIcon, label: "Favorites and settings" },
] as const;

const MOCK_ROWS = [
  { logo: "/hub/profile/logo/qwen.png", name: "Qwen3-14B", quant: "Q4_K_M" },
  {
    logo: "/hub/profile/logo/google.png",
    name: "gemma-4-E4B-it",
    quant: "UD-Q4_K_XL",
    active: true,
  },
  {
    logo: "/hub/profile/logo/deepseek.svg",
    name: "DeepSeek-R1",
    quant: "Q8_0",
  },
] as const;

/** Below the new-chat composer: a card that opens a dialog about the composer model picker. */
export const ModelPickerAnnouncement: FC<{ className?: string }> = ({
  className,
}) => {
  const source = useComposerModelPickerSource();
  const dismissed = useSyncExternalStore(subscribe, readDismissed, () => true);
  const [open, setOpen] = useState(false);
  const openPickerOnClose = useRef(false);

  // Chat only: Translate browses, and Compare and the API page own their selection.
  if (!source || source.browseOnly || source.selection) {
    return null;
  }

  return (
    <>
      {!dismissed && (
        <div
          className={cn(
            "flex justify-center px-2 fade-in slide-in-from-bottom-2 animate-in duration-500",
            className,
          )}
          data-model-picker-announcement=""
        >
          <div className="relative w-full max-w-[calc(460px*var(--ui-space-scale,1))]">
            <button
              type="button"
              onClick={() => setOpen(true)}
              className="group relative flex w-full items-center gap-4 overflow-hidden rounded-[calc(22px*var(--ui-space-scale,1))] py-5 pr-3 pl-5 text-left bg-white shadow-[var(--composer-shadow)] dark:bg-card transition-transform duration-200 hover:-translate-y-px focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <span className="relative flex min-w-0 flex-1 flex-col items-start gap-1">
                <span className="font-heading font-semibold text-base text-foreground tracking-[-0.01em]">
                  Meet the new model picker
                </span>
                <span className="inline-flex items-center gap-1 font-medium text-primary text-xs">
                  Learn more
                  <HugeiconsIcon
                    icon={ArrowRight01Icon}
                    strokeWidth={2}
                    className="size-3.5 transition-transform duration-200 group-hover:translate-x-0.5"
                  />
                </span>
              </span>
              <CardArt />
              <MascotImg
                src="Sloth emojis/sloth magnify final.png"
                aria-hidden="true"
                className="pointer-events-none absolute right-3 bottom-0 h-[calc(80px*var(--ui-space-scale,1))] w-auto translate-y-[14%] transition-transform duration-300 group-hover:translate-y-[6%]"
              />
            </button>
            <button
              type="button"
              onClick={dismiss}
              aria-label="Dismiss"
              className="absolute -top-2 -right-2 flex size-6 items-center justify-center rounded-full text-muted-foreground bg-white shadow-[var(--composer-shadow)] dark:bg-card transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <HugeiconsIcon
                icon={Cancel01Icon}
                strokeWidth={2.25}
                className="size-3"
              />
            </button>
          </div>
        </div>
      )}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          className="content-start gap-0 p-3 pb-6 sm:max-w-[calc(35rem*var(--ui-space-scale,1))]"
          onCloseAutoFocus={(event) => {
            if (!openPickerOnClose.current) {
              return;
            }
            openPickerOnClose.current = false;
            // The card is gone by now; focus belongs to the picker's search instead.
            event.preventDefault();
            source.onOpenChange(true);
          }}
        >
          <DialogHero />
          <div className="mt-7 flex flex-col items-center gap-2.5 px-4 text-center">
            <NewBadge />
            <DialogTitle className="text-balance font-heading font-semibold text-[1.65rem] leading-tight tracking-[-0.02em]">
              Meet the new model picker
            </DialogTitle>
            <DialogDescription className="text-balance">
              Models, quants and reasoning, right in the composer.
            </DialogDescription>
            <Button
              variant="primary"
              size="hero"
              className="group mt-2 gap-1.5"
              onClick={() => {
                openPickerOnClose.current = true;
                dismiss();
                setOpen(false);
              }}
            >
              Open model picker
              <HugeiconsIcon
                icon={ArrowRight01Icon}
                strokeWidth={2}
                className="size-4 transition-transform duration-200 group-hover:translate-x-0.5"
              />
            </Button>
          </div>
          <ul className="mt-7 grid grid-cols-2 gap-2 px-3 sm:grid-cols-4">
            {FEATURES.map((feature) => (
              <li
                key={feature.label}
                className="flex flex-col items-center gap-2.5 rounded-2xl bg-muted/50 px-2 py-4 text-center"
              >
                <span className="flex size-9 items-center justify-center rounded-xl bg-primary/12 text-primary">
                  <HugeiconsIcon
                    icon={feature.icon}
                    strokeWidth={1.9}
                    className="size-4.5"
                  />
                </span>
                <span className="text-balance font-medium text-foreground text-xs leading-snug">
                  {feature.label}
                </span>
              </li>
            ))}
          </ul>
        </DialogContent>
      </Dialog>
    </>
  );
};

const NewBadge: FC = () => (
  <span className="inline-flex items-center gap-1 rounded-full bg-primary/12 px-2 py-0.5 font-semibold text-primary text-ui-11 tracking-wide">
    <HugeiconsIcon icon={SparklesIcon} strokeWidth={2} className="size-3" />
    New
  </span>
);

const LogoDot: FC<{ src: string; className?: string }> = ({
  src,
  className,
}) => (
  <img
    src={publicAssetUrl(src)}
    alt=""
    draggable={false}
    className={cn("shrink-0 rounded-full bg-white object-contain", className)}
  />
);

/** A tilted mini picker with the magnifying-glass sloth peeking over it. */
const CardArt: FC = () => (
  <span
    aria-hidden="true"
    className="relative block h-[calc(92px*var(--ui-space-scale,1))] w-[calc(140px*var(--ui-space-scale,1))] shrink-0"
  >
    <span className="absolute top-1 left-0 flex w-[calc(118px*var(--ui-space-scale,1))] -rotate-3 flex-col gap-0.5 rounded-xl bg-muted p-1.5 transition-transform duration-300 group-hover:-rotate-1">
      {MOCK_ROWS.map((row) => (
        <span
          key={row.name}
          className={cn(
            "flex items-center gap-1.5 rounded-lg px-1.5 py-1",
            "active" in row && row.active && "bg-primary/12",
          )}
        >
          <LogoDot src={row.logo} className="size-3.5" />
          <span
            className={cn(
              "h-1.5 flex-1 rounded-full",
              "active" in row && row.active
                ? "bg-primary/70"
                : "bg-muted-foreground/25",
            )}
          />
          {"active" in row && row.active ? (
            <HugeiconsIcon
              icon={Tick02Icon}
              strokeWidth={2.5}
              className="size-3 text-primary"
            />
          ) : (
            <span className="size-3" />
          )}
        </span>
      ))}
    </span>
  </span>
);

/** The composer, its model chip, and the picker it opens, drawn with live theme colours. */
const DialogHero: FC = () => (
  <div
    aria-hidden="true"
    className="relative h-[calc(236px*var(--ui-space-scale,1))] overflow-hidden rounded-[calc(20px*var(--ui-space-scale,1))] bg-muted/50"
  >
    <MascotImg
      src="Sloth emojis/UnSloth Sparkling large.png"
      className="absolute bottom-[calc(64px*var(--ui-space-scale,1))] left-8 size-24 max-sm:right-5 max-sm:left-auto max-sm:size-16"
    />

    <div className="absolute top-7 right-16 flex w-[calc(250px*var(--ui-space-scale,1))] max-sm:right-auto max-sm:left-4 max-sm:w-[calc(224px*var(--ui-space-scale,1))] flex-col gap-1.5 rounded-2xl p-2 bg-white shadow-[var(--composer-shadow)] dark:bg-card">
      <div className="flex items-center gap-3 px-1.5 pt-0.5 text-muted-foreground">
        <HugeiconsIcon icon={StarIcon} className="size-3.5" />
        <HugeiconsIcon icon={Compass01Icon} className="size-3.5" />
        <span className="relative text-primary">
          <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
          <span className="absolute inset-x-0 -bottom-1 h-0.5 rounded-full bg-primary" />
        </span>
        <HugeiconsIcon icon={CloudIcon} className="size-3.5" />
      </div>
      <div className="flex items-center gap-1.5 rounded-lg bg-muted px-2 py-1 text-muted-foreground text-ui-11">
        <HugeiconsIcon icon={Search01Icon} className="size-3" />
        Search on device models
      </div>
      <div className="flex flex-col gap-0.5">
        {MOCK_ROWS.map((row) => (
          <div
            key={row.name}
            className={cn(
              "flex items-center gap-2 rounded-lg px-1.5 py-1 text-ui-11",
              "active" in row && row.active
                ? "bg-primary/12 text-foreground"
                : "text-foreground/80",
            )}
          >
            <LogoDot src={row.logo} className="size-4" />
            <span className="min-w-0 flex-1 truncate font-medium">
              {row.name}
            </span>
            <span className="rounded-md bg-muted px-1 font-mono text-muted-foreground text-ui-11">
              {row.quant}
            </span>
            {"active" in row && row.active ? (
              <HugeiconsIcon
                icon={Tick02Icon}
                strokeWidth={2.5}
                className="size-3 text-primary"
              />
            ) : (
              <span className="size-3" />
            )}
          </div>
        ))}
      </div>
    </div>

    <div className="absolute inset-x-4 bottom-4 flex h-[calc(52px*var(--ui-space-scale,1))] items-end justify-between rounded-2xl px-3.5 pb-2.5 bg-white shadow-[var(--composer-shadow)] dark:bg-card">
      <span className="absolute top-2.5 left-4 text-muted-foreground text-xs">
        Ask anything
      </span>
      <span />
      <span className="flex items-center gap-1.5 rounded-full bg-primary/12 py-0.5 pr-2 pl-1 font-medium text-foreground text-ui-11">
        <LogoDot src="/hub/profile/logo/google.png" className="size-4" />
        gemma-4-E4B-it
        <span className="text-muted-foreground">Thinking</span>
        <HugeiconsIcon
          icon={ArrowDown01Icon}
          strokeWidth={2}
          className="size-3 text-muted-foreground"
        />
      </span>
    </div>
  </div>
);
