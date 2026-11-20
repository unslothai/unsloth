// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MascotImg } from "@/components/mascot-img";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  PROFILE_TEXT_MAX_LENGTH,
  useUserProfileStore,
} from "@/features/profile/stores/user-profile-store";
import { type TranslationKey, useT } from "@/i18n";
import { Copy01Icon } from "@/lib/copy-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  ChipIcon,
  MessageCircleIcon,
  TestTubeOutlineIcon,
} from "@/lib/hugeicons-derived";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  AiMagicIcon,
  ChefHatIcon,
  CommandLineIcon,
  CpuIcon,
  CursorInfo02Icon,
  DashboardCircleIcon,
  SquareLock02Icon,
  WifiDisconnected01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { type StartChoice, greetingName } from "../lib/pages";
import type { DeviceSummary } from "../use-device-summary";
import {
  ChosenRing,
  Group,
  Mascot,
  Row,
  StepHeader,
  Tile,
} from "./onboarding-ui";

const WELCOME_POINTS: {
  icon: IconSvgElement;
  title: TranslationKey;
  body: TranslationKey;
}[] = [
  {
    icon: SquareLock02Icon,
    title: "onboarding.welcome.privateTitle",
    body: "onboarding.welcome.privateBody",
  },
  {
    icon: WifiDisconnected01Icon,
    title: "onboarding.welcome.offlineTitle",
    body: "onboarding.welcome.offlineBody",
  },
  {
    icon: TestTubeOutlineIcon,
    title: "onboarding.welcome.trainTitle",
    body: "onboarding.welcome.trainBody",
  },
];

export function WelcomeStep() {
  const t = useT();
  return (
    <div className="my-auto">
      <Mascot src="Sloth emojis/large sloth wave.png" className="size-28" />
      <div className="mt-6 mb-9 space-y-2.5 text-center">
        <h2 className="font-heading text-[2.25rem] leading-[1.1] font-semibold tracking-[-0.025em]">
          {t("onboarding.welcome.title")}
        </h2>
        <p className="text-muted-foreground text-[0.9375rem]">
          {t("onboarding.welcome.body")}
        </p>
      </div>
      <Group>
        {WELCOME_POINTS.map((point) => (
          <Row
            key={point.title}
            lead={<Tile icon={point.icon} />}
            label={t(point.title)}
            description={t(point.body)}
          />
        ))}
      </Group>
    </div>
  );
}

export function ProfileStep({ onSubmit }: { onSubmit: () => void }) {
  const t = useT();
  const displayName = useUserProfileStore((s) => s.displayName);
  const setDisplayName = useUserProfileStore((s) => s.setDisplayName);
  const name = greetingName(displayName);
  return (
    <div>
      <StepHeader
        title={t("onboarding.profile.title")}
        body={t("onboarding.profile.body")}
      />
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit();
        }}
      >
        <Input
          id="onboarding-name"
          aria-label={t("onboarding.profile.label")}
          autoFocus={true}
          autoComplete="given-name"
          maxLength={PROFILE_TEXT_MAX_LENGTH}
          placeholder={t("onboarding.profile.placeholder")}
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          className="h-12 px-5 text-center text-base"
        />
      </form>
      <div className="bg-muted/60 mt-6 flex flex-col items-center gap-3 rounded-2xl px-6 py-9">
        <MascotImg
          src="Sloth emojis/large sloth wave.png"
          className="size-14 object-contain"
        />
        <span className="font-heading max-w-full truncate text-[1.375rem] font-semibold tracking-[-0.01em]">
          {name
            ? t("welcome.goodToSeeYouName", { name })
            : t("welcome.goodToSeeYou")}
        </span>
        <span className="text-muted-foreground text-xs">
          {t("onboarding.profile.preview")}
        </span>
      </div>
    </div>
  );
}

function formatGb(value: number): string {
  return value >= 10
    ? Math.round(value).toString()
    : value.toFixed(1).replace(/\.0$/, "");
}

function Status({ ready }: { ready: boolean }) {
  const t = useT();
  return ready ? (
    <span className="flex items-center gap-1 text-xs font-medium text-emerald-600 dark:text-emerald-500">
      <HugeiconsIcon icon={Tick02Icon} className="size-3.5" strokeWidth={2.5} />
      {t("onboarding.device.ready")}
    </span>
  ) : (
    <span className="text-muted-foreground text-xs">
      {t("onboarding.device.unavailable")}
    </span>
  );
}

type ReadySummary = Extract<DeviceSummary, { status: "ready" }>;

function memoryLine(summary: ReadySummary, t: ReturnType<typeof useT>): string {
  const parts: string[] = [];
  if (summary.unifiedMemory && summary.ramGb !== null) {
    parts.push(
      t("onboarding.device.unified", { size: formatGb(summary.ramGb) }),
    );
  } else {
    if (summary.gpuName && summary.vramGb !== null) {
      parts.push(
        t("onboarding.device.vram", { size: formatGb(summary.vramGb) }),
      );
    }
    if (summary.ramGb !== null) {
      parts.push(t("onboarding.device.ram", { size: formatGb(summary.ramGb) }));
    }
  }
  return parts.join(" · ");
}

function AcceleratorRow({ summary }: { summary: DeviceSummary }) {
  const t = useT();
  if (summary.status === "checking") {
    return (
      <Row
        lead={
          <span className="flex size-9 shrink-0 items-center justify-center">
            <span className="border-muted-foreground/30 border-t-foreground size-4.5 animate-spin rounded-full border-2" />
          </span>
        }
        label={t("onboarding.device.checking")}
        muted={true}
      />
    );
  }
  if (summary.status === "error") {
    return (
      <Row
        lead={<Tile icon={CpuIcon} />}
        label={t("onboarding.device.error")}
        muted={true}
      />
    );
  }
  const name =
    summary.gpuName ??
    (summary.appleSilicon
      ? t("onboarding.device.appleSilicon")
      : t("onboarding.device.cpuOnly"));
  return (
    <Row
      lead={
        <Tile
          icon={summary.gpuName || summary.appleSilicon ? ChipIcon : CpuIcon}
          tone="accent"
        />
      }
      label={
        <>
          {name}
          {summary.gpuCount > 1 ? (
            <span className="text-muted-foreground font-normal">
              {" "}
              {t("onboarding.device.moreGpus", { count: summary.gpuCount - 1 })}
            </span>
          ) : null}
        </>
      }
      description={memoryLine(summary, t) || undefined}
    />
  );
}

export function DeviceStep({ summary }: { summary: DeviceSummary }) {
  const t = useT();
  const ready = summary.status === "ready" ? summary : null;
  return (
    <div>
      <StepHeader
        title={t("onboarding.device.title")}
        body={t("onboarding.device.body")}
      />
      <Group>
        <div aria-live="polite">
          <AcceleratorRow summary={summary} />
        </div>
        {ready ? (
          <>
            <Row
              lead={<Tile icon={MessageCircleIcon} />}
              label={t("onboarding.device.chat")}
              trailing={<Status ready={true} />}
            />
            <Row
              lead={<Tile icon={TestTubeOutlineIcon} />}
              label={t("onboarding.device.train")}
              muted={!ready.train}
              trailing={<Status ready={ready.train} />}
            />
            <Row
              lead={<Tile icon={AiMagicIcon} />}
              label={t("onboarding.device.media")}
              muted={!ready.media}
              trailing={<Status ready={ready.media} />}
            />
          </>
        ) : null}
      </Group>
      {ready && !(ready.train && ready.media) ? (
        <p className="text-muted-foreground mt-4 text-center text-xs">
          {t("onboarding.device.unavailableHint")}
        </p>
      ) : null}
    </div>
  );
}

const START_OPTIONS: {
  value: StartChoice;
  icon: IconSvgElement;
  title: TranslationKey;
  body: TranslationKey;
}[] = [
  {
    value: "chat",
    icon: MessageCircleIcon,
    title: "onboarding.start.chatTitle",
    body: "onboarding.start.chatBody",
  },
  {
    value: "train",
    icon: TestTubeOutlineIcon,
    title: "onboarding.start.trainTitle",
    body: "onboarding.start.trainBody",
  },
  {
    value: "recipes",
    icon: ChefHatIcon,
    title: "onboarding.start.recipesTitle",
    body: "onboarding.start.recipesBody",
  },
  {
    value: "hub",
    icon: DashboardCircleIcon,
    title: "onboarding.start.hubTitle",
    body: "onboarding.start.hubBody",
  },
];

function StartCard({
  option,
  active,
  disabled,
  onPick,
}: {
  option: (typeof START_OPTIONS)[number];
  active: boolean;
  disabled: boolean;
  onPick: () => void;
}) {
  const t = useT();
  return (
    <button
      type="button"
      aria-pressed={active}
      disabled={disabled}
      onClick={onPick}
      className={cn(
        "bg-card relative flex flex-col items-start gap-4 rounded-2xl p-5 text-left ring-1 ring-border transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
        disabled ? "cursor-not-allowed opacity-50" : "hover:bg-muted/60",
      )}
    >
      <span className="flex w-full items-start justify-between">
        <Tile icon={option.icon} tone={active ? "accent" : "plain"} />
        <span
          aria-hidden="true"
          className={cn(
            "flex size-4.5 items-center justify-center rounded-full border-[1.5px] transition-colors",
            active ? "border-control-accent" : "border-border",
          )}
        >
          <span
            className={cn(
              "bg-control-accent size-2 rounded-full transition-transform",
              active ? "scale-100" : "scale-0",
            )}
          />
        </span>
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold">{t(option.title)}</span>
        <span className="text-muted-foreground mt-1 block text-xs leading-relaxed">
          {disabled ? t("onboarding.start.trainUnavailable") : t(option.body)}
        </span>
      </span>
      <ChosenRing on={active} />
    </button>
  );
}

export function StartStep({
  value,
  onChange,
  trainAvailable,
}: {
  value: StartChoice;
  onChange: (next: StartChoice) => void;
  trainAvailable: boolean;
}) {
  const t = useT();
  return (
    <div>
      <StepHeader
        title={t("onboarding.start.title")}
        body={t("onboarding.start.body")}
      />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {START_OPTIONS.map((option) => (
          <StartCard
            key={option.value}
            option={option}
            active={value === option.value}
            disabled={option.value === "train" && !trainAvailable}
            onPick={() => onChange(option.value)}
          />
        ))}
      </div>
    </div>
  );
}

const AGENT_COMMAND = "unsloth start claude";

function CopyCommand() {
  const t = useT();
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
  return (
    <div className="bg-background flex items-center gap-2 rounded-xl py-1.5 pr-1.5 pl-3.5 shadow-xs">
      <code className="min-w-0 flex-1 truncate font-mono text-xs">
        {AGENT_COMMAND}
      </code>
      <button
        type="button"
        onClick={() => {
          void copyToClipboard(AGENT_COMMAND).then((ok) => {
            if (!ok) return;
            setCopied(true);
            if (timer.current !== null) window.clearTimeout(timer.current);
            timer.current = window.setTimeout(() => setCopied(false), 1600);
          });
        }}
        className="text-muted-foreground hover:bg-muted hover:text-foreground flex h-7 items-center gap-1.5 rounded-lg px-2 text-xs transition-colors"
      >
        <HugeiconsIcon
          icon={copied ? Tick02Icon : Copy01Icon}
          className="size-3.5"
        />
        {copied ? t("onboarding.done.copied") : t("onboarding.done.copy")}
      </button>
    </div>
  );
}

export function DoneStep({
  start,
  place,
  tour,
  onTourChange,
}: {
  start: StartChoice;
  place: string;
  tour: boolean;
  onTourChange: (next: boolean) => void;
}) {
  const t = useT();
  const displayName = useUserProfileStore((s) => s.displayName);
  const name = greetingName(displayName);
  return (
    <div className="my-auto">
      <Mascot src="Sloth emojis/large sloth yay.png" className="size-20" />
      <div className="mt-5 mb-8 space-y-2 text-center">
        <h2 className="font-heading text-[1.75rem] leading-[1.15] font-semibold tracking-[-0.02em]">
          {name
            ? t("onboarding.done.titleName", { name })
            : t("onboarding.done.title")}
        </h2>
        <p className="text-muted-foreground text-[0.9375rem]">
          {start === "chat"
            ? t("onboarding.done.startChat")
            : t("onboarding.done.startOther", { place })}
        </p>
      </div>
      <Group>
        <Row
          lead={<Tile icon={CursorInfo02Icon} />}
          label={
            <span id="onboarding-tour-label">{t("onboarding.done.tour")}</span>
          }
          description={t("onboarding.done.tourBody", { place })}
          trailing={
            <Switch
              aria-labelledby="onboarding-tour-label"
              checked={tour}
              onCheckedChange={onTourChange}
            />
          }
        />
        <Row
          lead={<Tile icon={CommandLineIcon} />}
          label={t("onboarding.done.agents")}
          description={t("onboarding.done.agentsBody")}
          below={<CopyCommand />}
        />
      </Group>
    </div>
  );
}
