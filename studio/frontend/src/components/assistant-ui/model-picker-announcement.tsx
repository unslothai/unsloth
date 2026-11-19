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
import { usePlatformStore } from "@/config/env";
import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
} from "@/features/auth";
import {
  formatParamCount,
  hfApiToken,
  resolveOwnerProviderLogo,
  useHfTokenStore,
  useOnlineStatus,
} from "@/features/hub";
import { useHubFeed } from "@/features/hub/hooks/use-hub-feed";
import { browserAccountKey } from "@/lib/account-transition";
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { MicIcon } from "@/lib/mic-icon";
import { cn } from "@/lib/utils";
import {
  Brain03Icon,
  Cancel01Icon,
  CloudIcon,
  Download01Icon,
  FilterMailIcon,
  GridViewIcon,
  Navigation05Icon,
  PlusSignIcon,
  Search01Icon,
  Settings01Icon,
  SparklesIcon,
  StarIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowUpIcon } from "lucide-react";
import {
  type FC,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

// Per account, so each user on a shared browser gets the card once. Listed in
// ACCOUNT_KEYED_PREFIXES so sign-ins keep it, and kept out of Reset preferences:
// only the X dismisses it, for good.
export const MODEL_PICKER_INTRO_DISMISSED_PREFIX =
  "unsloth_model_picker_intro_dismissed:";

function dismissedKey(): string {
  return MODEL_PICKER_INTRO_DISMISSED_PREFIX + browserAccountKey();
}

const listeners = new Set<() => void>();

function readDismissed(): boolean {
  try {
    return localStorage.getItem(dismissedKey()) === "true";
  } catch {
    return true;
  }
}

function dismiss(): void {
  try {
    localStorage.setItem(dismissedKey(), "true");
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
      event.key === null ||
      event.key.startsWith(MODEL_PICKER_INTRO_DISMISSED_PREFIX)
    ) {
      listener();
    }
  };
  window.addEventListener("storage", onStorage);
  window.addEventListener(AUTH_SESSION_STORED_EVENT, listener);
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(AUTH_SESSION_STORED_EVENT, listener);
    window.removeEventListener(AUTH_SESSION_CLEARED_EVENT, listener);
  };
}

const FEATURES = [
  { icon: GridViewIcon, label: "Easy browsing and switching" },
  { icon: Settings01Icon, label: "Run settings in reach" },
  { icon: Brain03Icon, label: "Reasoning built in" },
  { icon: StarIcon, label: "Star your favorites" },
] as const;

const MOCK_ROWS = [
  { logo: "/hub/profile/logo/qwen.png", name: "Qwen3-14B", quant: "Q4_K_M" },
  {
    logo: "/provider-logos/google-g.svg",
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
            // Focus belongs to the picker's search, not back on the card.
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
              className="mt-2 gap-1.5 px-5.5 py-2.5 text-[0.95rem]"
              onClick={() => {
                openPickerOnClose.current = true;
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
                className="flex items-center gap-2.5 rounded-full bg-muted/50 py-3 pr-4 pl-4"
              >
                <span className="flex shrink-0 items-center justify-center text-primary">
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
    className={cn("shrink-0 object-contain", className)}
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
const GOOGLE_MARK = "/provider-logos/google-g.svg";

interface HeroRow {
  name: string;
  logo: string;
  detail: string;
}

// Offline, or before the Hub answers: Trending Now's leaders at the time of writing.
const FALLBACK_ROWS: readonly HeroRow[] = [
  {
    name: "Qwen3.8-27B-GGUF",
    logo: "/hub/profile/logo/qwen.png",
    detail: "27B",
  },
  { name: "gemma-4-26B-A4B-it-GGUF", logo: GOOGLE_MARK, detail: "26B" },
  {
    name: "DeepSeek-V4.1-Flash-1017-GGUF",
    logo: "/hub/profile/logo/deepseek.svg",
    detail: "748B",
  },
];

// The top trending chat model leads; then the top Google and DeepSeek ones, so one maker never fills the list.
// Provider-logo ids, each with its offline stand-in.
const FEATURED_MAKERS = [
  ["google", 1],
  ["deepseek-ai", 2],
] as const;
const NON_CHAT_TASKS = new Set([
  "feature-extraction",
  "sentence-similarity",
  "text-to-image",
  "image-to-image",
  "text-to-video",
  "text-to-speech",
  "automatic-speech-recognition",
]);

/** From Model Hub's Trending Now (same feed and cache). */
function useTrendingRows(): readonly HeroRow[] {
  const online = useOnlineStatus();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const accessToken = hfApiToken(useHfTokenStore((s) => s.token));
  const { trending } = useHubFeed({
    accessToken,
    online,
    enabled: true,
    deviceType,
  });
  return useMemo(() => {
    const candidates = trending.results.flatMap((model) => {
      const [owner, repo] = model.id.split("/");
      const mark = resolveOwnerProviderLogo(owner, repo);
      if (!(mark && repo) || NON_CHAT_TASKS.has(model.pipelineTag ?? "")) {
        return [];
      }
      const row: HeroRow = {
        name: repo,
        logo: mark.id === "google" ? GOOGLE_MARK : mark.logoPath,
        detail: model.totalParams ? formatParamCount(model.totalParams) : "",
      };
      return [{ maker: mark.id, row }];
    });
    const lead = candidates[0];
    const rows = lead ? [lead.row] : [];
    for (const [maker, fallback] of FEATURED_MAKERS) {
      if (lead?.maker !== maker) {
        rows.push(
          candidates.find((c) => c.maker === maker)?.row ??
            FALLBACK_ROWS[fallback],
        );
      }
    }
    return rows.length >= 3 ? rows.slice(0, 3) : FALLBACK_ROWS;
  }, [trending.results]);
}

/** The composer with its chip open, drawn with the real picker's and composer's icons and surfaces. */
const DialogHero: FC = () => {
  const rows = useTrendingRows();
  const chosen = rows[0];
  return (
    <div
      aria-hidden="true"
      className="relative mx-auto mt-2 h-[calc(240px*var(--ui-space-scale,1))] max-sm:h-[calc(254px*var(--ui-space-scale,1))] w-full max-w-[calc(460px*var(--ui-space-scale,1))]"
    >
      <div className="absolute inset-x-0 bottom-4 flex flex-col gap-3 rounded-[calc(22px*var(--ui-space-scale,1))] bg-white px-3 pt-3 pb-2.5 shadow-[var(--composer-shadow)] dark:bg-card">
        <span className="px-1.5 text-muted-foreground text-xs">
          Ask anything
        </span>
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <HugeiconsIcon
            icon={PlusSignIcon}
            className="size-4.5 shrink-0 stroke-[1.75px]"
          />
          <span className="flex-1" />
          <span className="flex min-w-0 items-center gap-1.5 whitespace-nowrap rounded-full bg-muted-foreground/10 py-1.5 pr-3 pl-2 font-medium text-foreground text-sm">
            <LogoDot src={chosen.logo} className="size-4" />
            <span className="min-w-0 truncate">{chosen.name}</span>
            <span className="shrink-0 text-muted-foreground">Thinking</span>
          </span>
          <MicIcon className="mx-1 size-4.5 shrink-0" />
          <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
            <ArrowUpIcon className="size-4 stroke-2" />
          </span>
        </div>
      </div>

      <div className="absolute top-0 right-10 z-10 flex w-[calc(264px*var(--ui-space-scale,1))] max-w-[calc(100%-2.5rem)] flex-col gap-2 rounded-2xl bg-white p-2 shadow-[var(--popup-shadow)] dark:bg-card">
        <div className="flex items-center gap-3.5 px-2 pt-1 text-muted-foreground">
          <HugeiconsIcon
            icon={StarIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          <span className="relative text-foreground">
            <HugeiconsIcon
              icon={Navigation05Icon}
              strokeWidth={1.75}
              className="size-3.5"
            />
            <span className="absolute inset-x-0 -bottom-1.5 h-0.5 rounded-full bg-primary" />
          </span>
          <HugeiconsIcon
            icon={Download01Icon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          <HugeiconsIcon
            icon={CloudIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          <LogoDot src="/provider-logos/google-g.svg" className="size-3.5" />
          <LogoDot src="/hub/profile/logo/qwen.png" className="size-3.5" />
        </div>
        <div className="flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1.5 text-muted-foreground text-ui-11">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={1.75}
            className="size-3"
          />
          <span className="flex-1">Search models to download</span>
          <HugeiconsIcon
            icon={FilterMailIcon}
            strokeWidth={1.75}
            className="size-3"
          />
        </div>
        <div className="flex flex-col">
          {rows.map((row, index) => (
            <div
              key={row.name}
              className={cn(
                "flex items-center gap-2 rounded-lg px-2 py-1.5 text-foreground text-ui-11",
                index === 0 && "bg-muted-foreground/10",
              )}
            >
              <LogoDot src={row.logo} className="size-4" />
              <span className="min-w-0 flex-1 truncate">{row.name}</span>
              {row.detail ? (
                <span className="rounded-md bg-muted px-1.5 font-mono text-muted-foreground text-ui-11">
                  {row.detail}
                </span>
              ) : null}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
