// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { publicAssetUrl } from "@/components/mascot-img";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { UserAvatar } from "@/features/profile/components/user-avatar";
import {
  PROFILE_TEXT_MAX_LENGTH,
  useUserProfileStore,
} from "@/features/profile/stores/user-profile-store";
import { initialsFromName } from "@/features/profile/utils/avatar-initials";
import { type TranslationKey, useT } from "@/i18n";
import { Copy01Icon } from "@/lib/copy-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { ChipIcon, TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  CommandLineIcon,
  CpuIcon,
  CursorInfo02Icon,
  SquareLock02Icon,
  WifiDisconnected01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { motion } from "motion/react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { greetingName } from "../lib/pages";
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

const AVATARS = [
  "Sloth emojis/large sloth wave.png",
  "Sloth emojis/large sloth yay.png",
  "Sloth emojis/large sloth heart.png",
  "Sloth emojis/large sloth thumbs.png",
  "Sloth emojis/large sloth glasses.png",
  "Sloth emojis/large sloth cheeky.png",
  "Sloth emojis/sloth headphones.png",
  "Sloth emojis/UnSloth Sparkling large.png",
];

function AvatarChoice({
  active,
  label,
  onPick,
  children,
}: {
  active: boolean;
  label: string;
  onPick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      aria-label={label}
      title={label}
      onClick={onPick}
      className="bg-background relative flex aspect-square items-center justify-center overflow-hidden rounded-full ring-1 ring-border outline-none transition-transform hover:scale-105 focus-visible:ring-2 focus-visible:ring-ring"
    >
      {children}
      <ChosenRing on={active} className="ring-foreground ring-inset" />
    </button>
  );
}

export function ProfileStep({ onSubmit }: { onSubmit: () => void }) {
  const t = useT();
  const displayName = useUserProfileStore((s) => s.displayName);
  const setDisplayName = useUserProfileStore((s) => s.setDisplayName);
  const avatar = useUserProfileStore((s) => s.avatarDataUrl);
  const setAvatar = useUserProfileStore((s) => s.setAvatarDataUrl);
  const name = greetingName(displayName);
  return (
    <div>
      <StepHeader
        title={t("onboarding.profile.title")}
        body={t("onboarding.profile.body")}
      />
      <div className="flex flex-col items-center">
        <motion.div
          key={avatar ?? "initials"}
          initial={{ scale: 0.9, opacity: 0.6 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 380, damping: 22 }}
        >
          <UserAvatar
            name={displayName.trim() || "?"}
            imageUrl={avatar}
            size="lg"
            className="size-20 text-2xl ring-4 ring-background"
          />
        </motion.div>
        <p className="font-heading mt-3 max-w-full truncate text-lg font-semibold tracking-[-0.01em]">
          {name
            ? t("welcome.goodToSeeYouName", { name })
            : t("welcome.goodToSeeYou")}
        </p>
      </div>
      <form
        className="mt-5"
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
      <Group className="mt-5">
        <Row
          label={t("onboarding.profile.avatar")}
          description={t("onboarding.profile.avatarBody")}
          below={
            <div className="grid grid-cols-9 gap-2">
              <AvatarChoice
                active={avatar === null}
                label={t("onboarding.profile.initials")}
                onPick={() => setAvatar(null)}
              >
                <span className="text-muted-foreground text-xs font-semibold">
                  {initialsFromName(displayName.trim() || "?")}
                </span>
              </AvatarChoice>
              {AVATARS.map((path) => {
                const url = publicAssetUrl(path);
                return (
                  <AvatarChoice
                    key={path}
                    active={avatar === url}
                    label={t("onboarding.profile.sloth")}
                    onPick={() => setAvatar(url)}
                  >
                    <img
                      alt=""
                      src={url}
                      draggable={false}
                      className="size-full object-cover"
                    />
                  </AvatarChoice>
                );
              })}
            </div>
          }
        />
      </Group>
    </div>
  );
}

function formatGb(value: number): string {
  return value >= 10
    ? Math.round(value).toString()
    : value.toFixed(1).replace(/\.0$/, "");
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
      below={<Capabilities summary={summary} />}
    />
  );
}

const CAPABILITIES: {
  key: "chat" | "train" | "media";
  label: TranslationKey;
}[] = [
  { key: "chat", label: "onboarding.device.chat" },
  { key: "train", label: "onboarding.device.train" },
  { key: "media", label: "onboarding.device.media" },
];

function Capabilities({ summary }: { summary: ReadySummary }) {
  const t = useT();
  const ready = { chat: true, train: summary.train, media: summary.media };
  return (
    <div className="flex flex-wrap gap-1.5">
      {CAPABILITIES.map((c) => (
        <span
          key={c.key}
          className={cn(
            "bg-background flex h-7 items-center gap-1.5 rounded-full px-2.5 text-xs shadow-xs",
            ready[c.key]
              ? "text-foreground"
              : "text-muted-foreground opacity-70",
          )}
        >
          {ready[c.key] ? (
            <HugeiconsIcon
              icon={Tick02Icon}
              className="size-3 text-emerald-600 dark:text-emerald-500"
              strokeWidth={2.5}
            />
          ) : null}
          {t(c.label)}
        </span>
      ))}
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
  device,
  tour,
  onTourChange,
}: {
  device: DeviceSummary;
  tour: boolean;
  onTourChange: (next: boolean) => void;
}) {
  const t = useT();
  const displayName = useUserProfileStore((s) => s.displayName);
  const name = greetingName(displayName);
  return (
    <div className="my-auto">
      <Mascot src="Sloth emojis/large sloth yay.png" className="size-18" />
      <div className="mt-4 mb-7 space-y-2 text-center">
        <h2 className="font-heading text-[1.75rem] leading-[1.15] font-semibold tracking-[-0.02em]">
          {name
            ? t("onboarding.done.titleName", { name })
            : t("onboarding.done.title")}
        </h2>
        <p className="text-muted-foreground text-[0.9375rem]">
          {t("onboarding.done.startChat")}
        </p>
      </div>
      <Group>
        <div aria-live="polite">
          <AcceleratorRow summary={device} />
        </div>
        <Row
          lead={<Tile icon={CursorInfo02Icon} />}
          label={
            <span id="onboarding-tour-label">{t("onboarding.done.tour")}</span>
          }
          description={t("onboarding.done.tourBody")}
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
