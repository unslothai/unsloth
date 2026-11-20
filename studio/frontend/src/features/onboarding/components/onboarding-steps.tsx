// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MascotImg } from "@/components/mascot-img";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  PROFILE_TEXT_MAX_LENGTH,
  useUserProfileStore,
} from "@/features/profile/stores/user-profile-store";
import {
  COLOR_THEMES,
  type ColorThemeId,
  FLAVOR_THEME_IDS,
  type ThemeModeColors,
} from "@/features/settings/lib/color-themes";
import {
  type Theme,
  usePalette,
  useTheme,
} from "@/features/settings/stores/theme-store";
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
  CpuIcon,
  DashboardCircleIcon,
  LaptopIcon,
  Moon02Icon,
  SquareLock02Icon,
  Sun03Icon,
  WifiDisconnected01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { motion } from "motion/react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { type StartChoice, greetingName } from "../lib/pages";
import type { DeviceSummary } from "../use-device-summary";

export function StepHeader({ title, body }: { title: string; body: string }) {
  return (
    <div className="space-y-1.5">
      <h2 className="font-heading text-[1.6rem] leading-tight font-semibold tracking-[-0.01em]">
        {title}
      </h2>
      <p className="text-muted-foreground text-sm">{body}</p>
    </div>
  );
}

function Mascot({ src, className }: { src: string; className?: string }) {
  return (
    <motion.div
      aria-hidden="true"
      initial={{ scale: 0.85, rotate: -6, opacity: 0 }}
      animate={{ scale: 1, rotate: 0, opacity: 1 }}
      transition={{ type: "spring", stiffness: 260, damping: 18, delay: 0.05 }}
      className={className}
    >
      <MascotImg src={src} className="size-full object-contain" />
    </motion.div>
  );
}

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
    <div className="my-auto flex flex-col items-center pb-2 text-center">
      <Mascot src="Sloth emojis/large sloth wave.png" className="size-28" />
      <h2 className="font-heading mt-4 text-[1.9rem] leading-tight font-semibold tracking-[-0.015em]">
        {t("onboarding.welcome.title")}
      </h2>
      <p className="text-muted-foreground mt-1.5 text-sm">
        {t("onboarding.welcome.body")}
      </p>
      <ul className="mt-7 w-full max-w-sm space-y-3.5 text-left">
        {WELCOME_POINTS.map((point, i) => (
          <motion.li
            key={point.title}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.32,
              delay: 0.18 + i * 0.07,
              ease: [0.16, 1, 0.3, 1],
            }}
            className="flex items-start gap-3"
          >
            <span className="bg-primary/10 text-primary flex size-8 shrink-0 items-center justify-center rounded-full">
              <HugeiconsIcon
                icon={point.icon}
                className="size-4"
                strokeWidth={1.75}
              />
            </span>
            <span className="min-w-0">
              <span className="block text-sm font-medium">
                {t(point.title)}
              </span>
              <span className="text-muted-foreground block text-xs">
                {t(point.body)}
              </span>
            </span>
          </motion.li>
        ))}
      </ul>
    </div>
  );
}

export function ProfileStep({ onSubmit }: { onSubmit: () => void }) {
  const t = useT();
  const displayName = useUserProfileStore((s) => s.displayName);
  const setDisplayName = useUserProfileStore((s) => s.setDisplayName);
  const name = greetingName(displayName);
  return (
    <div className="space-y-6">
      <StepHeader
        title={t("onboarding.profile.title")}
        body={t("onboarding.profile.body")}
      />
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit();
        }}
        className="space-y-2"
      >
        <label htmlFor="onboarding-name" className="text-sm font-medium">
          {t("onboarding.profile.label")}
        </label>
        <Input
          id="onboarding-name"
          autoFocus={true}
          autoComplete="given-name"
          maxLength={PROFILE_TEXT_MAX_LENGTH}
          placeholder={t("onboarding.profile.placeholder")}
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          className="h-11 text-base"
        />
      </form>
      <div className="space-y-2">
        <p className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
          {t("onboarding.profile.preview")}
        </p>
        <div className="bg-muted/50 flex items-center justify-center gap-3 rounded-2xl px-5 py-7">
          <MascotImg
            src="Sloth emojis/large sloth wave.png"
            className="size-12 object-contain"
          />
          <span className="font-heading truncate text-xl font-semibold tracking-[-0.01em]">
            {name
              ? t("welcome.goodToSeeYouName", { name })
              : t("welcome.goodToSeeYou")}
          </span>
        </div>
      </div>
    </div>
  );
}

const MODES: { value: Theme; label: TranslationKey; icon: IconSvgElement }[] = [
  { value: "light", label: "settings.appearance.theme.light", icon: Sun03Icon },
  { value: "dark", label: "settings.appearance.theme.dark", icon: Moon02Icon },
  {
    value: "system",
    label: "settings.appearance.theme.system",
    icon: LaptopIcon,
  },
];

// A flavour per family, so the row reads as a range rather than near-duplicates.
const SWATCHES: ColorThemeId[] = [
  "standard",
  "matcha",
  "blueberry",
  "cherry",
  "tangerine",
  "taro",
  "honey",
  "espresso",
  "neon-cyberpunk",
];

function WindowSketch({ colors }: { colors: ThemeModeColors }) {
  return (
    <div
      className="flex h-full w-full"
      style={{ backgroundColor: colors.background }}
    >
      <div
        className="h-full w-[28%] space-y-1 p-1.5"
        style={{
          backgroundColor:
            colors.surface ??
            `color-mix(in oklab, ${colors.foreground} 5%, ${colors.background})`,
        }}
      >
        {[70, 50, 60].map((w) => (
          <div
            key={w}
            className="h-1 rounded-full opacity-25"
            style={{ width: `${w}%`, backgroundColor: colors.foreground }}
          />
        ))}
      </div>
      <div className="flex flex-1 flex-col justify-end p-2">
        <div
          className="mb-1.5 ml-auto h-2 w-1/2 rounded-full"
          style={{ backgroundColor: colors.accent }}
        />
        <div
          className="h-3 rounded-full border"
          style={{
            borderColor: `color-mix(in oklab, ${colors.foreground} 18%, transparent)`,
          }}
        />
      </div>
    </div>
  );
}

export function LookStep() {
  const t = useT();
  const { theme, resolved, setTheme } = useTheme();
  const { palette, setPalette } = usePalette();
  const meta = COLOR_THEMES[palette];
  const more = FLAVOR_THEME_IDS.length + 1 - SWATCHES.length;
  const nameOf = (id: ColorThemeId) =>
    COLOR_THEMES[id].name ?? t("settings.appearance.palette.standard");
  return (
    <div className="space-y-6">
      <StepHeader
        title={t("onboarding.look.title")}
        body={t("onboarding.look.body")}
      />
      <div className="space-y-2.5">
        <p className="text-sm font-medium">{t("onboarding.look.mode")}</p>
        <div className="grid grid-cols-3 gap-3">
          {MODES.map((mode) => {
            const active = theme === mode.value;
            return (
              <button
                key={mode.value}
                type="button"
                aria-pressed={active}
                onClick={() => setTheme(mode.value)}
                className="group flex flex-col gap-2 text-left outline-none"
              >
                <span
                  className={cn(
                    "relative block aspect-[16/10] overflow-hidden rounded-xl ring-1 ring-border transition-[box-shadow,transform] duration-200 group-hover:-translate-y-0.5 group-focus-visible:ring-2 group-focus-visible:ring-ring",
                    active && "ring-2 ring-control-accent",
                  )}
                >
                  {mode.value === "system" ? (
                    <span className="flex h-full">
                      <span className="h-full w-1/2 overflow-hidden">
                        <span className="block h-full w-[200%]">
                          <WindowSketch colors={meta.light} />
                        </span>
                      </span>
                      <span className="h-full w-1/2 overflow-hidden">
                        <span className="-ml-[100%] block h-full w-[200%]">
                          <WindowSketch colors={meta.dark} />
                        </span>
                      </span>
                    </span>
                  ) : (
                    <WindowSketch colors={meta[mode.value]} />
                  )}
                </span>
                <span
                  className={cn(
                    "flex items-center gap-1.5 text-xs font-medium",
                    active ? "text-foreground" : "text-muted-foreground",
                  )}
                >
                  <HugeiconsIcon icon={mode.icon} className="size-3.5" />
                  {t(mode.label)}
                </span>
              </button>
            );
          })}
        </div>
      </div>
      <div className="space-y-2.5">
        <div className="flex items-baseline justify-between gap-3">
          <p className="text-sm font-medium">{t("onboarding.look.theme")}</p>
          <p className="text-muted-foreground truncate text-xs">
            {nameOf(palette)}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2.5">
          {SWATCHES.map((id) => {
            const colors = COLOR_THEMES[id][resolved];
            const active = palette === id;
            return (
              <button
                key={id}
                type="button"
                aria-pressed={active}
                aria-label={nameOf(id)}
                title={nameOf(id)}
                onClick={() => setPalette(id)}
                className={cn(
                  "flex size-9 items-center justify-center rounded-full border border-black/10 transition-transform duration-150 outline-none hover:scale-110 focus-visible:ring-2 focus-visible:ring-ring dark:border-white/15",
                  active &&
                    "ring-foreground ring-2 ring-offset-2 ring-offset-background",
                )}
                style={{ backgroundColor: colors.background }}
              >
                <span
                  className="size-4 rounded-full"
                  style={{ backgroundColor: colors.accent }}
                />
              </button>
            );
          })}
          <span className="text-muted-foreground ml-1 text-xs">
            {t("onboarding.look.moreThemes", { count: more })}
          </span>
        </div>
      </div>
    </div>
  );
}

function formatGb(value: number): string {
  return value >= 10
    ? Math.round(value).toString()
    : value.toFixed(1).replace(/\.0$/, "");
}

function CapabilityRow({
  icon,
  label,
  ready,
}: { icon: IconSvgElement; label: string; ready: boolean }) {
  const t = useT();
  return (
    <li className="flex items-center gap-3 py-2.5">
      <HugeiconsIcon
        icon={icon}
        className={cn(
          "size-4.5 shrink-0",
          ready ? "text-foreground" : "text-muted-foreground/70",
        )}
        strokeWidth={1.75}
      />
      <span
        className={cn(
          "min-w-0 flex-1 truncate text-sm",
          !ready && "text-muted-foreground",
        )}
      >
        {label}
      </span>
      {ready ? (
        <span className="flex items-center gap-1 text-xs font-medium text-emerald-600 dark:text-emerald-500">
          <HugeiconsIcon
            icon={Tick02Icon}
            className="size-3.5"
            strokeWidth={2.5}
          />
          {t("onboarding.device.ready")}
        </span>
      ) : (
        <span className="text-muted-foreground text-xs">
          {t("onboarding.device.unavailable")}
        </span>
      )}
    </li>
  );
}

export function DeviceStep({ summary }: { summary: DeviceSummary }) {
  const t = useT();
  return (
    <div className="space-y-6">
      <StepHeader
        title={t("onboarding.device.title")}
        body={t("onboarding.device.body")}
      />
      <div
        className="bg-muted/50 flex min-h-18 items-center gap-3.5 rounded-2xl px-4 py-3.5"
        aria-live="polite"
      >
        {summary.status === "checking" ? (
          <>
            <span className="border-muted-foreground/30 border-t-foreground size-5 shrink-0 animate-spin rounded-full border-2" />
            <span className="text-muted-foreground text-sm">
              {t("onboarding.device.checking")}
            </span>
          </>
        ) : summary.status === "error" ? (
          <span className="text-muted-foreground text-sm">
            {t("onboarding.device.error")}
          </span>
        ) : (
          <>
            <span className="bg-background flex size-10 shrink-0 items-center justify-center rounded-xl shadow-xs">
              <HugeiconsIcon
                icon={
                  summary.gpuName || summary.appleSilicon ? ChipIcon : CpuIcon
                }
                className="size-5"
                strokeWidth={1.75}
              />
            </span>
            <span className="min-w-0">
              <span className="block truncate text-sm font-medium">
                {summary.gpuName ??
                  (summary.appleSilicon
                    ? t("onboarding.device.appleSilicon")
                    : t("onboarding.device.cpuOnly"))}
                {summary.gpuCount > 1 ? (
                  <span className="text-muted-foreground font-normal">
                    {" "}
                    {t("onboarding.device.moreGpus", {
                      count: summary.gpuCount - 1,
                    })}
                  </span>
                ) : null}
              </span>
              <span className="text-muted-foreground block truncate text-xs">
                {[
                  summary.unifiedMemory && summary.ramGb !== null
                    ? t("onboarding.device.unified", {
                        size: formatGb(summary.ramGb),
                      })
                    : null,
                  !summary.unifiedMemory &&
                  summary.vramGb !== null &&
                  summary.gpuName
                    ? t("onboarding.device.vram", {
                        size: formatGb(summary.vramGb),
                      })
                    : null,
                  !summary.unifiedMemory && summary.ramGb !== null
                    ? t("onboarding.device.ram", {
                        size: formatGb(summary.ramGb),
                      })
                    : null,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            </span>
          </>
        )}
      </div>
      {summary.status === "ready" ? (
        <div>
          <ul className="divide-border/70 divide-y">
            <CapabilityRow
              icon={MessageCircleIcon}
              label={t("onboarding.device.chat")}
              ready={true}
            />
            <CapabilityRow
              icon={TestTubeOutlineIcon}
              label={t("onboarding.device.train")}
              ready={summary.train}
            />
            <CapabilityRow
              icon={AiMagicIcon}
              label={t("onboarding.device.media")}
              ready={summary.media}
            />
          </ul>
          {!summary.train || !summary.media ? (
            <p className="text-muted-foreground mt-2 text-xs">
              {t("onboarding.device.unavailableHint")}
            </p>
          ) : null}
        </div>
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
    <div className="space-y-6">
      <StepHeader
        title={t("onboarding.start.title")}
        body={t("onboarding.start.body")}
      />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {START_OPTIONS.map((option) => {
          const disabled = option.value === "train" && !trainAvailable;
          const active = value === option.value;
          return (
            <button
              key={option.value}
              type="button"
              aria-pressed={active}
              disabled={disabled}
              onClick={() => onChange(option.value)}
              className={cn(
                "flex items-start gap-3 rounded-2xl p-4 text-left ring-1 ring-border transition-[box-shadow,background-color] duration-150 outline-none focus-visible:ring-2 focus-visible:ring-ring",
                active
                  ? "bg-control-accent/[0.07] ring-2 ring-control-accent"
                  : "hover:bg-muted/50",
                disabled &&
                  "cursor-not-allowed opacity-55 hover:bg-transparent",
              )}
            >
              <span
                className={cn(
                  "flex size-9 shrink-0 items-center justify-center rounded-xl",
                  active
                    ? "bg-control-accent text-control-accent-foreground"
                    : "bg-muted text-foreground",
                )}
              >
                <HugeiconsIcon
                  icon={option.icon}
                  className="size-4.5"
                  strokeWidth={1.75}
                />
              </span>
              <span className="min-w-0">
                <span className="block text-sm font-medium">
                  {t(option.title)}
                </span>
                <span className="text-muted-foreground block text-xs leading-relaxed">
                  {disabled
                    ? t("onboarding.start.trainUnavailable")
                    : t(option.body)}
                </span>
              </span>
            </button>
          );
        })}
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
    <div className="bg-muted/50 flex items-center gap-2 rounded-xl py-1.5 pr-1.5 pl-3">
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
        className="text-muted-foreground hover:bg-background hover:text-foreground flex h-7 items-center gap-1.5 rounded-lg px-2 text-xs transition-colors"
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

function SummaryItem({ children }: { children: ReactNode }) {
  return (
    <li className="flex items-center gap-2.5 text-sm">
      <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-500">
        <HugeiconsIcon icon={Tick02Icon} className="size-3" strokeWidth={3} />
      </span>
      <span className="min-w-0 truncate">{children}</span>
    </li>
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
  const { theme } = useTheme();
  const { palette } = usePalette();
  const name = greetingName(displayName);
  const mode = MODES.find((m) => m.value === theme) ?? MODES[2];
  return (
    <div className="my-auto flex flex-col items-center text-center">
      <Mascot src="Sloth emojis/large sloth yay.png" className="size-20" />
      <h2 className="font-heading mt-3 text-[1.6rem] leading-tight font-semibold tracking-[-0.01em]">
        {t("onboarding.done.title")}
      </h2>
      <p className="text-muted-foreground mt-1 text-sm">
        {t("onboarding.done.body")}
      </p>
      <div className="mt-5 w-full max-w-sm space-y-4 text-left">
        <ul className="space-y-2.5">
          <SummaryItem>{t("onboarding.done.local")}</SummaryItem>
          {name ? (
            <SummaryItem>{t("onboarding.done.name", { name })}</SummaryItem>
          ) : null}
          <SummaryItem>
            {t("onboarding.done.look", {
              theme:
                COLOR_THEMES[palette].name ??
                t("settings.appearance.palette.standard"),
              mode: t(mode.label),
            })}
          </SummaryItem>
          <SummaryItem>
            {start === "chat"
              ? t("onboarding.done.startChat")
              : t("onboarding.done.startOther", { place })}
          </SummaryItem>
        </ul>
        <div className="flex items-center gap-3 px-0.5 py-1">
          <span className="min-w-0 flex-1">
            <span
              id="onboarding-tour-label"
              className="block text-sm font-medium"
            >
              {t("onboarding.done.tour")}
            </span>
            <span className="text-muted-foreground block text-xs">
              {t("onboarding.done.tourBody", { place })}
            </span>
          </span>
          <Switch
            aria-labelledby="onboarding-tour-label"
            checked={tour}
            onCheckedChange={onTourChange}
          />
        </div>
        <div className="space-y-1.5">
          <p className="text-muted-foreground text-xs">
            {t("onboarding.done.agents")}
          </p>
          <CopyCommand />
        </div>
      </div>
    </div>
  );
}
