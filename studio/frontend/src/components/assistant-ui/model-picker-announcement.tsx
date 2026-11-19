// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useComposerModelPickerSource } from "@/components/assistant-ui/composer-model-picker/context";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { useShortcutLabel } from "@/features/settings/hooks/use-shortcut";
import { cn } from "@/lib/utils";
import {
  Brain03Icon,
  Cancel01Icon,
  Download01Icon,
  GridViewIcon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type FC, useRef, useState, useSyncExternalStore } from "react";

// Literal copies live in Reset preferences (general-tab.tsx) and ACCOUNT_CHROME_KEYS.
export const MODEL_PICKER_ANNOUNCEMENT_DISMISSED_KEY =
  "unsloth_model_picker_announcement_dismissed";

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
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (
      event.key === MODEL_PICKER_ANNOUNCEMENT_DISMISSED_KEY ||
      event.key === null
    )
      listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

const FEATURES = [
  {
    icon: GridViewIcon,
    title: "Every model in one place",
    body: "Favorites, Discover models, On Device and Connected models, plus a tab for each maker and connection.",
  },
  {
    icon: Download01Icon,
    title: "Download and load in one pick",
    body: "Discover lists models recommended for this device. Pick a quant and it downloads and loads; models that will not fit are dimmed.",
  },
  {
    icon: Brain03Icon,
    title: "Reasoning beside the model",
    body: "Set the effort level or turn thinking on and off, and switch quantizations, right under the list.",
  },
  {
    icon: StarIcon,
    title: "Favorites and Run settings",
    body: "Star models and drag them into order. The gear opens Run settings without leaving the picker.",
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
  const shortcut = useShortcutLabel("openModelPicker");

  // Chat only: Translate browses, and Compare and the API page own their selection.
  if (!source || source.browseOnly || source.selection) return null;

  return (
    <>
      {!dismissed && (
        <div
          className={cn(
            "flex justify-center px-2 fade-in slide-in-from-bottom-1 animate-in duration-300",
            className,
          )}
          data-model-picker-announcement=""
        >
          <div className="relative w-full max-w-[calc(400px*var(--ui-space-scale,1))]">
            <button
              type="button"
              onClick={() => setOpen(true)}
              className="flex w-full items-center gap-4 rounded-2xl bg-muted/40 py-3.5 pr-4 pl-5 text-left ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)] transition-colors hover:bg-muted/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <span className="flex min-w-0 flex-1 flex-col gap-1">
                <span className="font-heading font-medium text-foreground text-sm">
                  New model picker
                </span>
                <span className="text-muted-foreground text-xs leading-snug">
                  Pick, download and tune models right from the composer.{" "}
                  <span className="whitespace-nowrap font-medium text-foreground">
                    Learn more
                  </span>
                </span>
              </span>
              <CardArt className="h-[calc(52px*var(--ui-space-scale,1))] w-auto shrink-0 text-foreground/60" />
            </button>
            <button
              type="button"
              onClick={dismiss}
              aria-label="Dismiss"
              className="absolute -top-2 -right-2 flex size-6 items-center justify-center rounded-full bg-muted-foreground/80 text-background shadow-sm transition-colors hover:bg-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <HugeiconsIcon
                icon={Cancel01Icon}
                strokeWidth={2.5}
                className="size-3"
              />
            </button>
          </div>
        </div>
      )}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          className="gap-0 px-6 pt-6 pb-6 sm:max-w-[calc(30rem*var(--ui-space-scale,1))]"
          onCloseAutoFocus={(event) => {
            if (!openPickerOnClose.current) return;
            openPickerOnClose.current = false;
            // The card is gone by now; focus belongs to the picker's search instead.
            event.preventDefault();
            source.onOpenChange(true);
          }}
        >
          <DialogArt className="mx-auto mt-2 h-auto w-full max-w-[calc(300px*var(--ui-space-scale,1))] text-foreground/70" />
          <div className="mt-5 flex flex-col items-center gap-2 text-center">
            <span className="font-medium text-primary text-sm">New</span>
            <DialogTitle className="text-balance font-semibold text-2xl leading-tight tracking-[-0.01em]">
              Choose models from the composer
            </DialogTitle>
            <DialogDescription className="text-balance">
              Click the model chip in the composer to switch models, quants and
              reasoning without leaving the chat
              {shortcut ? (
                <>
                  , or press{" "}
                  <kbd className="rounded-md bg-muted px-1.5 py-0.5 font-mono text-foreground text-xs">
                    {shortcut}
                  </kbd>
                </>
              ) : null}
              .
            </DialogDescription>
            <Button
              variant="primary"
              size="hero"
              className="mt-3"
              onClick={() => {
                openPickerOnClose.current = true;
                dismiss();
                setOpen(false);
              }}
            >
              Open model picker
            </Button>
          </div>
          <ul className="mt-6 flex flex-col rounded-2xl bg-muted/50 px-4">
            {FEATURES.map((feature, index) => (
              <li
                key={feature.title}
                className={cn(
                  "flex items-start gap-3.5 py-3.5",
                  index > 0 && "border-border/60 border-t",
                )}
              >
                <HugeiconsIcon
                  icon={feature.icon}
                  strokeWidth={1.75}
                  className="mt-0.5 size-5 shrink-0 text-muted-foreground"
                />
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="font-medium text-foreground text-sm">
                    {feature.title}
                  </span>
                  <span className="text-muted-foreground text-xs leading-snug">
                    {feature.body}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </DialogContent>
      </Dialog>
    </>
  );
};

/** The picker panel and the composer chip, joined by a dotted line. */
const CardArt: FC<{ className?: string }> = ({ className }) => (
  <svg
    viewBox="0 0 120 60"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.25}
    strokeLinecap="round"
    className={className}
    aria-hidden="true"
  >
    <rect x="3" y="4" width="54" height="52" rx="6" />
    <circle cx="12" cy="16" r="2.5" />
    <path d="M18 16h28" />
    <rect
      x="7"
      y="23.5"
      width="46"
      height="13"
      rx="4"
      className="fill-primary/20 stroke-primary"
    />
    <circle cx="12" cy="30" r="2.5" className="stroke-primary" />
    <path d="M18 30h24" className="stroke-primary" />
    <circle cx="12" cy="44" r="2.5" />
    <path d="M18 44h20" />
    <path
      d="M58 30C72 30 78 34 80 40"
      strokeWidth={2}
      strokeDasharray="0 4"
      className="stroke-primary"
    />
    <rect x="72" y="42" width="45" height="15" rx="7.5" />
    <circle cx="80" cy="49.5" r="2.5" />
    <path d="M86 49.5h18" />
    <path d="M108 48l2.5 2.5L113 48" />
  </svg>
);

/** The composer with its model chip, and the picker panel it opens. */
const DialogArt: FC<{ className?: string }> = ({ className }) => (
  <svg
    viewBox="0 0 320 172"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    aria-hidden="true"
  >
    <rect x="22" y="6" width="156" height="104" rx="12" />
    <circle cx="38" cy="22" r="4" className="fill-primary stroke-primary" />
    <circle cx="52" cy="22" r="4" />
    <circle cx="66" cy="22" r="4" />
    <circle cx="80" cy="22" r="4" />
    <circle cx="94" cy="22" r="4" />
    <path d="M22 34h156" />
    <circle cx="38" cy="48" r="4.5" />
    <path d="M50 48h70" />
    <path d="M162 45l1.2 2.5 2.8.4-2 2 .5 2.8-2.5-1.3-2.5 1.3.5-2.8-2-2 2.8-.4z" />
    <rect
      x="28"
      y="59"
      width="144"
      height="18"
      rx="6"
      className="fill-primary/15 stroke-primary"
    />
    <circle cx="38" cy="68" r="4.5" className="stroke-primary" />
    <path d="M50 68h86" className="stroke-primary" />
    <circle cx="38" cy="90" r="4.5" />
    <path d="M50 90h56" />
    <path
      d="M184 58C226 58 252 84 252 128"
      strokeWidth={2.5}
      strokeDasharray="0 6"
      className="stroke-primary"
    />
    <rect x="8" y="118" width="304" height="50" rx="16" />
    <path d="M26 136h92" className="opacity-50" />
    <rect x="200" y="140" width="98" height="20" rx="10" />
    <circle cx="211" cy="150" r="4" />
    <path d="M220 150h52" />
    <path d="M281 148l3 3 3-3" />
  </svg>
);
