// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { publicAssetUrl } from "@/components/mascot-img";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { UserAvatar } from "@/features/profile/components/user-avatar";
import {
  PROFILE_TEXT_MAX_LENGTH,
  useUserProfileStore,
} from "@/features/profile/stores/user-profile-store";
import { initialsFromName } from "@/features/profile/utils/avatar-initials";
import { resizeImageFileToDataUrl } from "@/features/profile/utils/resize-image-file";
import { type TranslationKey, useT } from "@/i18n";
import { ChipIcon } from "@/lib/hugeicons-derived";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  CpuIcon,
  CursorInfo02Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion } from "motion/react";
import { type ReactNode, useRef, useState } from "react";
import { greetingName } from "../lib/pages";
import type { DeviceSummary } from "../use-device-summary";
import { ChosenRing, Group, Mascot, Row, Tile } from "./onboarding-ui";

const AVATARS = [
  "Sloth emojis/large sloth wave.png",
  "Sloth emojis/large sloth yay.png",
  "Sloth emojis/large sloth glasses.png",
  "Sloth emojis/sloth headphones.png",
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
      className="bg-muted relative flex size-11 shrink-0 items-center justify-center overflow-hidden rounded-full outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      {children}
      <ChosenRing on={active} className="ring-foreground ring-inset" />
    </button>
  );
}

export function WelcomeStep({ onSubmit }: { onSubmit: () => void }) {
  const t = useT();
  const displayName = useUserProfileStore((s) => s.displayName);
  const setDisplayName = useUserProfileStore((s) => s.setDisplayName);
  const avatar = useUserProfileStore((s) => s.avatarDataUrl);
  const setAvatar = useUserProfileStore((s) => s.setAvatarDataUrl);
  const name = greetingName(displayName);
  const fileInput = useRef<HTMLInputElement>(null);
  const [uploadError, setUploadError] = useState(false);
  const [uploading, setUploading] = useState(false);
  // Anything that is not one of the sloths here is the user's own picture (or a sloth picked in Settings).
  const custom =
    avatar !== null && !AVATARS.some((path) => publicAssetUrl(path) === avatar);
  const onUpload = async (file: File | undefined) => {
    if (!file) return;
    setUploadError(false);
    setUploading(true);
    try {
      setAvatar(await resizeImageFileToDataUrl(file));
    } catch {
      setUploadError(true);
    } finally {
      setUploading(false);
    }
  };
  return (
    <div>
      <div className="flex flex-col items-center text-center">
        <motion.div
          key={avatar ?? (displayName.trim() ? "initials" : "mascot")}
          initial={{ opacity: 0.4 }}
          animate={{ opacity: 1 }}
          transition={{ type: "spring", stiffness: 380, damping: 22 }}
        >
          {avatar === null && !displayName.trim() ? (
            <Mascot
              src="Sloth emojis/large sloth wave.png"
              className="size-24"
            />
          ) : (
            <UserAvatar
              name={displayName.trim() || "?"}
              imageUrl={avatar}
              size="lg"
              className="size-20 text-2xl ring-4 ring-background"
            />
          )}
        </motion.div>
        <h2 className="font-heading mt-6 max-w-full truncate text-[2rem] leading-[1.15] font-semibold tracking-[-0.025em]">
          {name
            ? t("onboarding.welcome.titleName", { name })
            : t("onboarding.welcome.title")}
        </h2>
        <p className="text-muted-foreground mt-3 text-[0.9375rem]">
          {t("onboarding.welcome.body")}
        </p>
      </div>
      <form
        className="mt-10"
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
      <div className="mt-6">
        <div className="flex items-center justify-center gap-3">
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
          {custom && avatar ? (
            <AvatarChoice
              active={true}
              label={t("onboarding.profile.yours")}
              onPick={() => fileInput.current?.click()}
            >
              <img
                alt=""
                src={avatar}
                draggable={false}
                className="size-full object-cover"
              />
            </AvatarChoice>
          ) : null}
          <input
            ref={fileInput}
            type="file"
            accept="image/jpeg,image/png,image/webp,image/gif"
            className="sr-only"
            tabIndex={-1}
            aria-hidden="true"
            onChange={(e) => {
              void onUpload(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>
        <div className="mt-5 flex justify-center">
          <Button
            type="button"
            variant="muted"
            className="h-10 gap-2 rounded-full px-5"
            disabled={uploading}
            onClick={() => fileInput.current?.click()}
          >
            <HugeiconsIcon
              icon={Upload01Icon}
              className="size-4"
              strokeWidth={1.75}
            />
            {uploading
              ? t("onboarding.profile.uploading")
              : t("onboarding.profile.upload")}
          </Button>
        </div>
        {uploadError ? (
          <p className="text-destructive mt-3 text-center text-xs">
            {t("onboarding.profile.uploadError")}
          </p>
        ) : null}
      </div>
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
            "bg-muted flex h-7 items-center gap-1.5 rounded-full px-2.5 text-xs",
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
      <div className="mx-auto flex size-14 items-center justify-center rounded-full bg-control-accent text-control-accent-foreground">
        <HugeiconsIcon icon={Tick02Icon} className="size-6" strokeWidth={2.5} />
      </div>
      <div className="mt-6 mb-10 space-y-2.5 text-center">
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
      </Group>
    </div>
  );
}
