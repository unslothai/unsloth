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
import { useShortcutLabel } from "@/features/settings/hooks/use-shortcut";
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
  {
    icon: GridViewIcon,
    title: "Every model in one place",
    body: "Favorites, Discover, On Device and Connected models, with a tab per maker and connection.",
  },
  {
    icon: Download01Icon,
    title: "One pick to download and run",
    body: "Discover shows models suited to this device. Choose a quant and it downloads and loads.",
  },
  {
    icon: Brain03Icon,
    title: "Reasoning beside the model",
    body: "Set the effort level, turn thinking on or off, and switch quantizations.",
  },
  {
    icon: StarIcon,
    title: "Favorites and Run settings",
    body: "Star models, drag them into order, and tune Run settings without leaving the picker.",
  },
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

const GLOW =
  "pointer-events-none absolute inset-0 bg-[radial-gradient(var(--glow-shape),color-mix(in_oklab,var(--primary)_var(--glow-strength),transparent),transparent_70%)]";

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
              className="group relative flex w-full items-center gap-4 overflow-hidden rounded-[calc(22px*var(--ui-space-scale,1))] bg-muted/50 py-4 pr-3 pl-5 text-left shadow-[0_1px_2px_rgb(0_0_0/0.04),0_8px_24px_-12px_rgb(0_0_0/0.18)] ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-edge-gain,1)),transparent)] transition-[background-color,box-shadow,transform] duration-200 hover:-translate-y-px hover:bg-muted/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <span
                className={GLOW}
                style={{
                  ["--glow-shape" as string]: "70% 120% at 100% 50%",
                  ["--glow-strength" as string]: "16%",
                }}
              />
              <span className="relative flex min-w-0 flex-1 flex-col items-start gap-1.5">
                <NewBadge />
                <span className="font-heading font-semibold text-[0.975rem] text-foreground tracking-[-0.01em]">
                  Meet the new model picker
                </span>
                <span className="text-muted-foreground text-xs leading-relaxed">
                  Pick, download and tune models right from the composer.
                </span>
                <span className="mt-0.5 inline-flex items-center gap-1 font-medium text-primary text-xs">
                  Learn more
                  <HugeiconsIcon
                    icon={ArrowRight01Icon}
                    strokeWidth={2}
                    className="size-3.5 transition-transform duration-200 group-hover:translate-x-0.5"
                  />
                </span>
              </span>
              <CardArt />
            </button>
            <button
              type="button"
              onClick={dismiss}
              aria-label="Dismiss"
              className="absolute -top-2 -right-2 flex size-6 items-center justify-center rounded-full bg-background text-muted-foreground shadow-sm ring-1 ring-border transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
          className="gap-0 p-3 pb-6 sm:max-w-[calc(35rem*var(--ui-space-scale,1))]"
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
          <div className="mt-6 flex flex-col items-center gap-2.5 px-4 text-center">
            <NewBadge label="New in Studio" />
            <DialogTitle className="text-balance font-heading font-semibold text-[1.65rem] leading-tight tracking-[-0.02em]">
              Meet the new model picker
            </DialogTitle>
            <DialogDescription className="max-w-[calc(26rem*var(--ui-space-scale,1))] text-balance leading-relaxed">
              Click the model chip in the composer to switch models, quants and
              reasoning without leaving the chat
              {shortcut ? (
                <>
                  , or press{" "}
                  <kbd className="rounded-md bg-muted px-1.5 py-0.5 font-mono text-foreground text-xs ring-1 ring-border/60">
                    {shortcut}
                  </kbd>
                </>
              ) : null}
              .
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
          <ul className="mt-6 grid gap-2 px-3 sm:grid-cols-2">
            {FEATURES.map((feature) => (
              <li
                key={feature.title}
                className="flex items-start gap-3 rounded-2xl bg-muted/50 p-3.5 ring-1 ring-border/40"
              >
                <span className="flex size-8 shrink-0 items-center justify-center rounded-xl bg-primary/12 text-primary">
                  <HugeiconsIcon
                    icon={feature.icon}
                    strokeWidth={1.9}
                    className="size-4"
                  />
                </span>
                <span className="flex min-w-0 flex-col gap-0.5">
                  <span className="font-medium text-foreground text-sm">
                    {feature.title}
                  </span>
                  <span className="text-muted-foreground text-xs leading-relaxed">
                    {feature.body}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </DialogContent>
      </Dialog>
    </>
  );
};

const NewBadge: FC<{ label?: string }> = ({ label = "New" }) => (
  <span className="inline-flex items-center gap-1 rounded-full bg-primary/12 px-2 py-0.5 font-semibold text-primary text-ui-11 tracking-wide">
    <HugeiconsIcon icon={SparklesIcon} strokeWidth={2} className="size-3" />
    {label}
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
    className={cn(
      "shrink-0 rounded-full bg-white object-contain ring-1 ring-black/5",
      className,
    )}
  />
);

/** A tilted mini picker with the magnifying-glass sloth peeking over it. */
const CardArt: FC = () => (
  <span
    aria-hidden="true"
    className="relative block h-[calc(92px*var(--ui-space-scale,1))] w-[calc(140px*var(--ui-space-scale,1))] shrink-0"
  >
    <span className="absolute top-1 left-0 flex w-[calc(118px*var(--ui-space-scale,1))] -rotate-3 flex-col gap-0.5 rounded-xl bg-background p-1.5 shadow-[0_6px_20px_-8px_rgb(0_0_0/0.35)] ring-1 ring-border/70 transition-transform duration-300 group-hover:-rotate-1">
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
    <MascotImg
      src="Sloth emojis/sloth magnify final.png"
      className="absolute -right-1 -bottom-1 size-16 drop-shadow-[0_4px_8px_rgb(0_0_0/0.25)] transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:rotate-3"
    />
  </span>
);

/** The composer, its model chip, and the picker it opens, drawn with live theme colours. */
const DialogHero: FC = () => (
  <div
    aria-hidden="true"
    className="relative h-[calc(236px*var(--ui-space-scale,1))] overflow-hidden rounded-[calc(20px*var(--ui-space-scale,1))] bg-muted/40 ring-1 ring-border/50"
  >
    <span
      className={GLOW}
      style={{
        ["--glow-shape" as string]: "90% 80% at 65% 0%",
        ["--glow-strength" as string]: "30%",
      }}
    />
    <span className="pointer-events-none absolute inset-0 bg-[radial-gradient(color-mix(in_oklab,var(--foreground)_14%,transparent)_1px,transparent_1px)] bg-size-[calc(16px*var(--ui-space-scale,1))_calc(16px*var(--ui-space-scale,1))] [mask-image:linear-gradient(to_bottom,black,transparent_85%)]" />

    <MascotImg
      src="Sloth emojis/UnSloth Sparkling large.png"
      className="absolute top-5 left-6 size-24 -rotate-6 drop-shadow-[0_8px_16px_rgb(0_0_0/0.25)] max-sm:top-auto max-sm:right-5 max-sm:bottom-[calc(62px*var(--ui-space-scale,1))] max-sm:left-auto max-sm:size-16 max-sm:rotate-6"
    />

    <div className="absolute top-7 right-16 flex w-[calc(250px*var(--ui-space-scale,1))] max-sm:right-auto max-sm:left-4 max-sm:w-[calc(224px*var(--ui-space-scale,1))] flex-col gap-1.5 rounded-2xl bg-background p-2 shadow-[0_18px_40px_-16px_rgb(0_0_0/0.45)] ring-1 ring-border/70">
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

    <div className="absolute inset-x-4 bottom-4 flex h-[calc(52px*var(--ui-space-scale,1))] items-end justify-between rounded-2xl bg-background px-3.5 pb-2.5 shadow-[0_10px_30px_-14px_rgb(0_0_0/0.4)] ring-1 ring-border/70">
      <span className="absolute top-2.5 left-4 text-muted-foreground text-xs">
        Ask anything
      </span>
      <span />
      <span className="flex items-center gap-1.5 rounded-full bg-primary/12 py-0.5 pr-2 pl-1 font-medium text-foreground text-ui-11 ring-1 ring-primary/50">
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
