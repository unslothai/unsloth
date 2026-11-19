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
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import {
  Brain03Icon,
  Cancel01Icon,
  CloudIcon,
  Compass01Icon,
  Download01Icon,
  FilterHorizontalIcon,
  GridViewIcon,
  PlusSignIcon,
  Search01Icon,
  SparklesIcon,
  StarIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowUpIcon, MicIcon } from "lucide-react";
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
              className="relative flex w-full items-center gap-4 overflow-hidden rounded-[calc(22px*var(--ui-space-scale,1))] py-5 pr-3 pl-5 text-left bg-white shadow-[var(--composer-shadow)] dark:bg-card focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <span className="relative flex min-w-0 flex-1 flex-col items-start gap-1">
                <span className="font-heading font-semibold text-base text-foreground tracking-[-0.01em]">
                  Meet the new model picker
                </span>
                <span className="inline-flex items-center gap-1 font-medium text-primary text-xs">
                  Learn more
                  <HugeiconsIcon
                    icon={ChevronRightStandardIcon}
                    strokeWidth={2}
                    className="size-3.5"
                  />
                </span>
              </span>
              <CardArt />
              <MascotImg
                src="Sloth emojis/sloth magnify final.png"
                aria-hidden="true"
                className="pointer-events-none absolute right-3 bottom-0 h-[calc(80px*var(--ui-space-scale,1))] w-auto"
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
          className="content-start gap-0 p-6 sm:max-w-[calc(35rem*var(--ui-space-scale,1))]"
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
              className="mt-2 gap-1"
              onClick={() => {
                openPickerOnClose.current = true;
                dismiss();
                setOpen(false);
              }}
            >
              Open model picker
              <HugeiconsIcon
                icon={ChevronRightStandardIcon}
                strokeWidth={2}
                className="size-4"
              />
            </Button>
          </div>
          <ul className="mt-7 grid grid-cols-2 gap-2">
            {FEATURES.map((feature) => (
              <li
                key={feature.label}
                className="flex items-center gap-2.5 rounded-full bg-muted/50 py-1.5 pr-4 pl-1.5"
              >
                <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary/12 text-primary">
                  <HugeiconsIcon
                    icon={feature.icon}
                    strokeWidth={1.9}
                    className="size-4.5"
                  />
                </span>
                <span className="font-medium text-foreground text-sm leading-snug">
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
    <span className="absolute top-1 left-0 flex w-[calc(118px*var(--ui-space-scale,1))] -rotate-3 flex-col gap-0.5 rounded-xl bg-muted p-1.5">
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
/** The composer with its chip open, styled like the real picker and composer. */
const DialogHero: FC = () => (
  <div
    aria-hidden="true"
    className="relative h-[calc(300px*var(--ui-space-scale,1))] overflow-hidden rounded-[calc(20px*var(--ui-space-scale,1))] bg-muted/50"
  >
    <div className="absolute top-5 right-6 flex w-[calc(264px*var(--ui-space-scale,1))] flex-col gap-2 rounded-2xl bg-white p-2 shadow-[var(--composer-shadow)] max-sm:w-[calc(236px*var(--ui-space-scale,1))] dark:bg-card">
      <div className="flex items-center gap-3.5 px-2 pt-1 text-muted-foreground">
        <HugeiconsIcon icon={StarIcon} className="size-3.5" />
        <HugeiconsIcon icon={Compass01Icon} className="size-3.5" />
        <span className="relative text-foreground">
          <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
          <span className="absolute inset-x-0 -bottom-1.5 h-0.5 rounded-full bg-primary" />
        </span>
        <HugeiconsIcon icon={CloudIcon} className="size-3.5" />
        <LogoDot src="/hub/profile/logo/google.png" className="size-3.5" />
        <LogoDot src="/hub/profile/logo/qwen.png" className="size-3.5" />
      </div>
      <div className="flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1.5 text-muted-foreground text-ui-11">
        <HugeiconsIcon icon={Search01Icon} className="size-3" />
        <span className="flex-1">Search on device models</span>
        <HugeiconsIcon icon={FilterHorizontalIcon} className="size-3" />
      </div>
      <div className="flex flex-col">
        {MOCK_ROWS.map((row) => (
          <div
            key={row.name}
            className={cn(
              "flex items-center gap-2 rounded-lg px-2 py-1.5 text-foreground text-ui-11",
              "active" in row && row.active && "bg-muted-foreground/10",
            )}
          >
            <LogoDot src={row.logo} className="size-4" />
            <span className="min-w-0 flex-1 truncate">{row.name}</span>
            {"active" in row && row.active ? (
              <span className="size-1.5 rounded-full bg-primary" />
            ) : null}
            <span className="rounded-full bg-muted px-1.5 font-mono text-muted-foreground text-ui-11">
              {row.quant}
            </span>
            <HugeiconsIcon
              icon={StarIcon}
              className="size-3 text-muted-foreground/60"
            />
          </div>
        ))}
      </div>
    </div>

    <div className="absolute inset-x-6 bottom-6 flex flex-col gap-3 rounded-[calc(22px*var(--ui-space-scale,1))] bg-white px-3 pt-3 pb-2.5 shadow-[var(--composer-shadow)] dark:bg-card">
      <span className="px-1.5 text-muted-foreground text-xs">Ask anything</span>
      <div className="flex items-center gap-1.5 text-muted-foreground">
        <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
        <span className="flex-1" />
        <span className="flex items-center gap-1.5 rounded-full bg-muted-foreground/10 py-1 pr-2.5 pl-1.5 font-medium text-foreground text-xs min-w-0 whitespace-nowrap">
          <LogoDot src="/hub/profile/logo/google.png" className="size-3.5" />
          <span className="min-w-0 truncate">gemma-4-E4B-it</span>
          <span className="shrink-0 text-muted-foreground">Thinking</span>
        </span>
        <MicIcon className="mx-1 size-4" />
        <span className="flex size-7 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <ArrowUpIcon className="size-4 stroke-2" />
        </span>
      </div>
    </div>
  </div>
);
