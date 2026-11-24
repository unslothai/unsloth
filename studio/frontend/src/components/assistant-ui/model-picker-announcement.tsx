// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useComposerModelPickerSource } from "@/components/assistant-ui/composer-model-picker/context";
import { MascotImg, publicAssetUrl } from "@/components/mascot-img";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
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
import { BulbIcon } from "@/lib/bulb-icon";
import {
  ChevronDownStandardIcon,
  ChevronLeftStandardIcon,
  ChevronRightStandardIcon,
  ChevronUpStandardIcon,
} from "@/lib/chevron-icons";
import { MicIcon } from "@/lib/mic-icon";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  CloudIcon,
  DashboardCircleIcon,
  Download01Icon,
  FilterMailIcon,
  MoreVerticalIcon,
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

// Per account, kept across sign-ins and out of Reset preferences: only the X hides it.
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

const featureIcon = (icon: typeof StarIcon) => (
  <HugeiconsIcon icon={icon} strokeWidth={1.9} className="size-4" />
);

const FEATURES = [
  {
    icon: featureIcon(DashboardCircleIcon),
    label: "Find and switch models fast",
  },
  { icon: featureIcon(Settings01Icon), label: "Run settings for each model" },
  {
    icon: <BulbIcon className="size-4" />,
    label: "Adjust thinking effort easily",
  },
  { icon: featureIcon(StarIcon), label: "Star the ones you use most" },
] as const;

const USE_STEPS = [
  "Browse Discover for models that fit your device",
  "Click the gear to tweak settings before loading",
  "Star models to keep them in Favorites",
  "When a model is loaded, click its row or the arrow to expand or collapse the picker",
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

export const ModelPickerAnnouncement: FC<{ className?: string }> = ({
  className,
}) => {
  const source = useComposerModelPickerSource();
  const dismissed = useSyncExternalStore(subscribe, readDismissed, () => true);
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState(0);
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
              onClick={() => {
                setPage(0);
                setOpen(true);
              }}
              className="glass-float-surface relative flex w-full items-center gap-4 overflow-hidden rounded-[calc(22px*var(--ui-space-scale,1))] py-5 pr-3 pl-5 text-left bg-white shadow-[var(--composer-shadow)] dark:bg-card focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
              className="glass-float-surface absolute -top-2 -right-2 flex size-6 items-center justify-center rounded-full text-muted-foreground bg-white shadow-[var(--composer-shadow)] dark:bg-card transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
          showCloseButton={false}
          className="content-start gap-0 p-6 sm:max-w-[calc(35rem*var(--ui-space-scale,1))]"
          onCloseAutoFocus={(event) => {
            if (!openPickerOnClose.current) {
              return;
            }
            openPickerOnClose.current = false;
            event.preventDefault();
            source.onOpenChange(true);
          }}
        >
          <DialogClose asChild={true}>
            <Button
              variant="ghost"
              size="icon-sm"
              className="absolute top-3 right-3 z-10 max-sm:fixed max-sm:top-[calc(0.75rem+var(--studio-window-chrome-top,0px))]"
            >
              <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} />
              <span className="sr-only">Close</span>
            </Button>
          </DialogClose>
          {page === 0 ? (
            <>
              <div className="mt-4 flex flex-col items-center gap-2.5 px-4 text-center">
                <NewBadge />
                <DialogTitle className="text-balance font-heading font-semibold text-[1.65rem] leading-tight tracking-[-0.02em]">
                  Meet the new model picker
                </DialogTitle>
                <DialogDescription className="text-balance">
                  The model picker now lives in the chat box, next to the send
                  button.
                </DialogDescription>
              </div>
              <div className="mt-8">
                <DialogHero />
              </div>
              <ul className="mx-auto mt-7 grid w-full max-w-[calc(488px*var(--ui-space-scale,1))] grid-cols-2 gap-2">
                {FEATURES.map((feature) => (
                  <li
                    key={feature.label}
                    className="flex items-center gap-2 rounded-full bg-muted dark:bg-muted/50 py-2.5 pr-3 pl-3.5"
                  >
                    <span className="flex shrink-0 items-center justify-center text-primary">
                      {feature.icon}
                    </span>
                    <span className="whitespace-nowrap font-medium text-foreground text-[0.84rem] max-sm:whitespace-normal leading-snug">
                      {feature.label}
                    </span>
                  </li>
                ))}
              </ul>
              <div className="mt-6 flex justify-center">
                <Button
                  variant="dark"
                  size="hero"
                  className="gap-1.5 px-5.5 py-2.5 text-[0.9rem]"
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
            </>
          ) : (
            <>
              <FoldHero />
              <div className="mt-7 flex flex-col items-center gap-2.5 px-4 text-center">
                <DialogTitle className="text-balance font-heading font-semibold text-[1.65rem] leading-tight tracking-[-0.02em]">
                  How to use it
                </DialogTitle>
                <DialogDescription className="sr-only">
                  Four ways to use the model picker.
                </DialogDescription>
              </div>
              <ol className="mt-6 flex flex-col gap-2.5 rounded-[calc(22px*var(--ui-space-scale,1))] bg-muted dark:bg-muted/50 px-5 py-4 text-sm">
                {USE_STEPS.map((step, index) => (
                  <li key={step} className="flex items-baseline gap-3">
                    <span className="w-3 shrink-0 font-semibold text-primary">
                      {index + 1}
                    </span>
                    <span className="text-foreground">{step}</span>
                  </li>
                ))}
              </ol>
              <Button
                variant="dark"
                size="hero"
                className="mt-6 gap-1.5 justify-self-center px-5.5 py-2.5 text-[0.9rem]"
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
            </>
          )}
          <div className="mt-6 flex items-center justify-between">
            <Button
              variant="ghost"
              className={cn("gap-1 text-[0.95rem]", page === 0 && "invisible")}
              onClick={() => setPage(0)}
            >
              <HugeiconsIcon
                icon={ChevronLeftStandardIcon}
                strokeWidth={2}
                className="size-4"
              />
              Back
            </Button>
            <div className="flex items-center">
              {[0, 1].map((dot) => (
                <button
                  key={dot}
                  type="button"
                  onClick={() => setPage(dot)}
                  aria-label={`Page ${dot + 1}`}
                  aria-current={dot === page ? "step" : undefined}
                  className="group/dot grid size-6 place-items-center rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <span
                    className={cn(
                      "size-2 rounded-full transition-colors",
                      dot === page
                        ? "bg-primary"
                        : "bg-muted-foreground/30 group-hover/dot:bg-muted-foreground/60",
                    )}
                  />
                </button>
              ))}
            </div>
            <Button
              variant="ghost"
              className={cn("gap-1 text-[0.95rem]", page === 1 && "invisible")}
              onClick={() => setPage(1)}
            >
              Next
              <HugeiconsIcon
                icon={ChevronRightStandardIcon}
                strokeWidth={2}
                className="size-4"
              />
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};

const NewBadge: FC = () => (
  <span className="inline-flex items-center gap-1 rounded-full bg-primary/12 px-2 py-0.5 font-bold text-primary text-ui-11 uppercase tracking-wide">
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

const GOOGLE_MARK = "/provider-logos/google-g.svg";

interface HeroRow {
  name: string;
  logo: string;
  detail: string;
}

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

// [provider-logo id, FALLBACK_ROWS index]
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
          <span className="flex size-7 shrink-0 items-center justify-center text-foreground">
            <HugeiconsIcon
              icon={PlusSignIcon}
              className="size-4.5 stroke-[1.75px]"
            />
          </span>
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
                "flex items-center gap-2 rounded-full px-2.5 py-1.5 text-foreground text-ui-11",
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

const MOCK_SURFACE =
  "flex shrink-0 flex-col gap-2 rounded-xl bg-white p-2 shadow-[var(--composer-shadow)] dark:bg-card";

const EFFORT_STOPS = [0, 0.25, 0.5, 0.75, 1];

const EffortMock: FC = () => (
  <div className="flex flex-col gap-1.5 px-2 pb-1">
    <div className="flex items-center justify-between text-ui-11">
      <span className="font-medium text-muted-foreground tracking-wide">
        EFFORT
      </span>
      <span className="text-foreground">Medium</span>
    </div>
    <div className="relative h-3">
      <span className="composer-effort-track absolute inset-x-0 top-[calc(1.5px*var(--ui-space-scale,1))] h-[calc(9px*var(--ui-space-scale,1))] rounded-full" />
      <span className="composer-effort-fill absolute top-[calc(1.5px*var(--ui-space-scale,1))] left-0 h-[calc(9px*var(--ui-space-scale,1))] w-[calc(50%+4.5px*var(--ui-space-scale,1))] rounded-full" />
      {EFFORT_STOPS.map((stop) =>
        stop === 0.5 ? null : (
          <span
            key={stop}
            style={{
              left: `calc(4.5px * var(--ui-space-scale, 1) + ${stop} * (100% - 9px * var(--ui-space-scale, 1)))`,
            }}
            className={cn(
              "absolute top-1/2 size-0.5 -translate-x-1/2 -translate-y-1/2 rounded-full",
              stop < 0.5 ? "composer-effort-dot-on" : "composer-effort-dot-off",
            )}
          />
        ),
      )}
      <span className="composer-effort-thumb absolute top-0 left-1/2 size-3 -translate-x-1/2 rounded-full" />
    </div>
  </div>
);

const FoldHero: FC = () => {
  const rows = useTrendingRows();
  const model = rows[0];
  return (
    <div
      aria-hidden="true"
      className="mt-9 flex items-center justify-center gap-3 max-sm:flex-col"
    >
      <div
        className={cn(MOCK_SURFACE, "w-[calc(216px*var(--ui-space-scale,1))]")}
      >
        <div className="flex items-center gap-2 rounded-full bg-muted-foreground/10 py-1 pr-1 pl-2.5 text-foreground text-ui-11">
          <LogoDot src={model.logo} className="size-4" />
          <span className="min-w-0 flex-1 truncate">{model.name}</span>
          <span className="size-1.5 shrink-0 rounded-full bg-primary" />
          <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-muted-foreground/15">
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={2}
              className="size-3.5 text-primary"
            />
          </span>
        </div>
        <EffortMock />
      </div>
      <HugeiconsIcon
        icon={ChevronRightStandardIcon}
        strokeWidth={2}
        className="size-5 shrink-0 text-muted-foreground max-sm:rotate-90"
      />
      <div
        className={cn(MOCK_SURFACE, "w-[calc(250px*var(--ui-space-scale,1))]")}
      >
        <div className="flex items-center gap-3 pl-2 text-muted-foreground">
          <HugeiconsIcon
            icon={StarIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          <HugeiconsIcon
            icon={Navigation05Icon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          <span className="relative text-foreground">
            <HugeiconsIcon
              icon={Download01Icon}
              strokeWidth={1.75}
              className="size-3.5"
            />
            <span className="absolute inset-x-0 -bottom-1.5 h-0.5 rounded-full bg-primary" />
          </span>
          <HugeiconsIcon
            icon={CloudIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          <span className="flex-1" />
          <span className="flex size-6 shrink-0 -translate-y-0.5 items-center justify-center rounded-full bg-muted-foreground/15">
            <HugeiconsIcon
              icon={ChevronUpStandardIcon}
              strokeWidth={2}
              className="size-3.5 text-primary"
            />
          </span>
        </div>
        <div className="flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1.5 text-muted-foreground text-ui-11">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={1.75}
            className="size-3"
          />
          <span className="flex-1">Search on device models</span>
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
                "flex items-center gap-2 rounded-full px-2.5 py-1.5 text-foreground text-ui-11",
                index === 0 && "bg-muted-foreground/10",
              )}
            >
              <LogoDot src={row.logo} className="size-4" />
              <span className="min-w-0 flex-1 truncate">{row.name}</span>
              {index === 0 ? (
                <>
                  <span className="size-1.5 shrink-0 rounded-full bg-primary" />
                  <HugeiconsIcon
                    icon={MoreVerticalIcon}
                    strokeWidth={2}
                    className="size-3.5 shrink-0 text-primary"
                  />
                  <HugeiconsIcon
                    icon={Settings01Icon}
                    strokeWidth={1.75}
                    className="size-3.5 shrink-0 text-primary"
                  />
                  <HugeiconsIcon
                    icon={StarIcon}
                    strokeWidth={1.75}
                    className="size-3.5 shrink-0 text-primary"
                  />
                </>
              ) : null}
            </div>
          ))}
        </div>
        <div className="border-border border-t pt-1.5">
          <EffortMock />
        </div>
      </div>
    </div>
  );
};
