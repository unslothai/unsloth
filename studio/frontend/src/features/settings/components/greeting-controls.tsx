// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MascotImg } from "@/components/mascot-img";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  GREETING_ICON_LIMITS,
  GREETING_IMAGE_LIMITS,
  GREETING_IMAGE_TYPES,
  GREETING_TEXT_MAX_LENGTH,
  isGreetingDataUrl,
  resizeImageFileToDataUrl,
  useUserProfileStore,
  validGreetingUpload,
} from "@/features/profile";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  SmileIcon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useId, useRef, useState } from "react";
import { SettingsRow } from "./settings-row";

type UploadField = "greetingIconDataUrl" | "greetingImageDataUrl";
type UploadError = "file" | "image";
type GreetingMode = "default" | "custom" | "image" | "off";

const PROFILE_STORAGE_KEY = "unsloth_user_profile";
const ACCEPT = GREETING_IMAGE_TYPES.join(",");
const MODES: { mode: GreetingMode; label: TranslationKey }[] = [
  { mode: "default", label: "settings.profile.greetingDefault" },
  { mode: "custom", label: "settings.profile.greetingCustom" },
  { mode: "image", label: "settings.profile.greetingImage" },
  { mode: "off", label: "settings.profile.noneLabel" },
];
// "Off" keeps writing showGreetingSloth, so the old switch's value maps straight onto a mode.
const MODE_PATCH = {
  default: {
    showGreetingSloth: true,
    greetingStyle: "text",
    greetingIcon: "sloth",
  },
  custom: {
    showGreetingSloth: true,
    greetingStyle: "text",
    greetingIcon: "custom",
  },
  image: { showGreetingSloth: true, greetingStyle: "image" },
  off: { showGreetingSloth: false },
} as const;
const SLOTH_PREVIEW = "Sloth emojis/large sloth wave.png";
// A type specimen, not a word, so the tiles need no translation.
const SAMPLE_TEXT = "Aa";

// A full localStorage quota drops the write silently; check it landed before claiming success.
function persistedValue(field: UploadField): unknown {
  try {
    const raw = window.localStorage.getItem(PROFILE_STORAGE_KEY);
    return raw
      ? (JSON.parse(raw) as { state?: Record<string, unknown> }).state?.[field]
      : null;
  } catch {
    return null;
  }
}

function useGreetingMode(): GreetingMode {
  const on = useUserProfileStore((s) => s.showGreetingSloth);
  const style = useUserProfileStore((s) => s.greetingStyle);
  const icon = useUserProfileStore((s) => s.greetingIcon);
  if (!on) {
    return "off";
  }
  if (style === "image") {
    return "image";
  }
  return icon === "custom" ? "custom" : "default";
}

function useGreetingUpload() {
  const setGreeting = useUserProfileStore((s) => s.setGreeting);
  const [error, setError] = useState<UploadError | null>(null);
  const [busy, setBusy] = useState(false);
  const upload = async (
    field: UploadField,
    file: File | undefined,
  ): Promise<void> => {
    if (!file) {
      return;
    }
    setError(null);
    if (!validGreetingUpload(file)) {
      setError("file");
      return;
    }
    setBusy(true);
    try {
      const dataUrl = await resizeImageFileToDataUrl(
        file,
        field === "greetingIconDataUrl"
          ? GREETING_ICON_LIMITS
          : GREETING_IMAGE_LIMITS,
      );
      if (!isGreetingDataUrl(dataUrl)) {
        throw new Error("encode");
      }
      const previous = useUserProfileStore.getState()[field];
      let stored = false;
      try {
        setGreeting({ [field]: dataUrl });
        stored = persistedValue(field) === dataUrl;
      } catch {
        // Handled below: a throwing quota write is the same failure as a dropped one.
      }
      if (!stored) {
        setGreeting({ [field]: previous });
        throw new Error("storage");
      }
    } catch {
      setError("image");
    } finally {
      setBusy(false);
    }
  };
  return { upload, error, busy };
}

function UploadControls({
  field,
  preview,
}: {
  field: UploadField;
  preview: string | null;
}) {
  const t = useT();
  const setGreeting = useUserProfileStore((s) => s.setGreeting);
  const { upload, error, busy } = useGreetingUpload();
  const input = useRef<HTMLInputElement>(null);
  return (
    <div className="flex flex-col items-end gap-1.5">
      <div className="flex items-center gap-2">
        <input
          ref={input}
          className="sr-only"
          tabIndex={-1}
          aria-label={`${t("settings.profile.greetingSloth")}: ${t("chatBackground.upload")}`}
          type="file"
          accept={ACCEPT}
          disabled={busy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            upload(field, file);
          }}
        />
        {preview && (
          <img
            src={preview}
            alt=""
            className="h-8 max-w-24 shrink-0 object-contain"
            data-greeting-preview=""
          />
        )}
        {preview && (
          <Button
            className="text-foreground"
            type="button"
            variant="ghost"
            size="sm"
            disabled={busy}
            onClick={() => setGreeting({ [field]: null })}
          >
            <HugeiconsIcon
              icon={Cancel01Icon}
              strokeWidth={1.75}
              className="size-4"
              aria-hidden="true"
            />
            {t("settings.profile.removePhoto")}
          </Button>
        )}
        <Button
          className="text-foreground"
          type="button"
          variant="outline"
          size="sm"
          disabled={busy}
          onClick={() => input.current?.click()}
        >
          <HugeiconsIcon
            icon={Upload01Icon}
            strokeWidth={1.75}
            className="size-4"
            aria-hidden="true"
          />
          {busy
            ? t("common.loading")
            : t(preview ? "chatBackground.replace" : "chatBackground.upload")}
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-xs text-destructive">
          {t(
            error === "file"
              ? "settings.profile.greetingFileError"
              : "settings.profile.greetingImageError",
          )}
        </p>
      )}
    </div>
  );
}

// The controls for the chosen mode, shown under the picker in the same row.
function GreetingModeOptions({ mode }: { mode: GreetingMode }) {
  const t = useT();
  const setGreeting = useUserProfileStore((s) => s.setGreeting);
  const text = useUserProfileStore((s) => s.greetingText);
  const iconDataUrl = useUserProfileStore((s) => s.greetingIconDataUrl);
  const imageDataUrl = useUserProfileStore((s) => s.greetingImageDataUrl);
  if (mode === "custom") {
    return (
      <div className="flex flex-col gap-3">
        <UploadControls field="greetingIconDataUrl" preview={iconDataUrl} />
        <Input
          aria-label={t("settings.profile.greetingText")}
          value={text}
          maxLength={GREETING_TEXT_MAX_LENGTH}
          placeholder={t("welcome.onYourMind")}
          onChange={(event) =>
            setGreeting({ greetingText: event.target.value })
          }
        />
        <p className="text-right text-xs text-muted-foreground">
          {t("settings.profile.greetingTextDescription")}
        </p>
      </div>
    );
  }
  if (mode === "image") {
    return (
      <UploadControls field="greetingImageDataUrl" preview={imageDataUrl} />
    );
  }
  return null;
}

// A small landscape in the accent colour, so an empty Image tile still reads as "a picture".
function ImagePlaceholder() {
  const id = useId();
  return (
    <svg
      viewBox="0 0 64 36"
      aria-hidden="true"
      className="h-8 w-14 overflow-hidden rounded-[5px]"
    >
      <defs>
        <linearGradient id={`${id}-sky`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="var(--primary)" stopOpacity="0.32" />
          <stop offset="1" stopColor="var(--primary)" stopOpacity="0.08" />
        </linearGradient>
      </defs>
      <rect width="64" height="36" fill={`url(#${id}-sky)`} />
      <circle cx="47" cy="11" r="5" fill="var(--primary)" fillOpacity="0.55" />
      <path
        d="M0 36 L18 17 L30 28 L38 21 L64 36 Z"
        fill="var(--primary)"
        fillOpacity="0.45"
      />
      <path
        d="M0 36 L12 27 L22 33 L36 24 L52 31 L64 26 L64 36 Z"
        fill="var(--primary)"
        fillOpacity="0.8"
      />
    </svg>
  );
}

function GreetingTiles({ mode }: { mode: GreetingMode }) {
  const t = useT();
  const setGreeting = useUserProfileStore((s) => s.setGreeting);
  const iconDataUrl = useUserProfileStore((s) => s.greetingIconDataUrl);
  const imageDataUrl = useUserProfileStore((s) => s.greetingImageDataUrl);
  const art: Record<GreetingMode, ReactNode> = {
    default: (
      <span className="flex items-center gap-1.5 text-sm font-medium">
        <MascotImg src={SLOTH_PREVIEW} className="size-5" />
        {SAMPLE_TEXT}
      </span>
    ),
    custom: (
      <span className="flex items-center gap-1.5 text-sm font-medium">
        {iconDataUrl ? (
          <img src={iconDataUrl} alt="" className="size-5 object-contain" />
        ) : (
          <HugeiconsIcon
            icon={SmileIcon}
            strokeWidth={1.75}
            className="size-5"
          />
        )}
        {SAMPLE_TEXT}
      </span>
    ),
    image: imageDataUrl ? (
      <img
        src={imageDataUrl}
        alt=""
        className="max-h-8 max-w-[80%] object-contain"
      />
    ) : (
      <ImagePlaceholder />
    ),
    off: null,
  };
  return (
    <div
      role="radiogroup"
      aria-label={t("settings.profile.greetingSloth")}
      className="grid w-[calc(24rem*var(--ui-space-scale,1))] max-w-full grid-cols-4 gap-2"
    >
      {MODES.map((m) => (
        <button
          key={m.mode}
          type="button"
          role="radio"
          aria-checked={mode === m.mode}
          onClick={() => setGreeting(MODE_PATCH[m.mode])}
          className="group flex cursor-pointer flex-col items-center gap-1.5"
        >
          <span
            className={cn(
              "flex h-14 w-full items-center justify-center rounded-lg border bg-background text-foreground transition-colors",
              mode === m.mode
                ? "border-primary"
                : "border-border group-hover:border-[color-mix(in_srgb,var(--foreground)_calc(30%*var(--contrast-wash-gain,1)),transparent)]",
              m.mode === "off" && "border-dashed",
            )}
          >
            {art[m.mode]}
          </span>
          <span
            className={cn(
              "text-xs",
              mode === m.mode ? "text-foreground" : "text-muted-foreground",
            )}
          >
            {t(m.label)}
          </span>
        </button>
      ))}
    </div>
  );
}

// Every greeting option lives in this one row: tiles beside the title that preview each mode,
// then the chosen mode's upload or text underneath.
export function GreetingSettingsRows() {
  const t = useT();
  const mode = useGreetingMode();
  return (
    <SettingsRow
      label={t("settings.profile.greetingSloth")}
      description={t("settings.profile.greetingSlothDescription")}
      wideControl={true}
      below={
        mode === "custom" || mode === "image" ? (
          <div className="flex w-[calc(24rem*var(--ui-space-scale,1))] max-w-full flex-col gap-3">
            <GreetingModeOptions mode={mode} />
          </div>
        ) : undefined
      }
    >
      <GreetingTiles mode={mode} />
    </SettingsRow>
  );
}
