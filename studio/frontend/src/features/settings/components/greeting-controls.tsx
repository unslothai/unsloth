// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
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
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { MenuTickIcon } from "@/lib/tick-icon";
import { Upload01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRef, useState } from "react";
import { SettingsRow } from "./settings-row";

type UploadField = "greetingIconDataUrl" | "greetingImageDataUrl";
type UploadError = "file" | "image";
type GreetingMode = "unsloth" | "customIcon" | "textOnly" | "image" | "off";

const PROFILE_STORAGE_KEY = "unsloth_user_profile";
const ACCEPT = GREETING_IMAGE_TYPES.join(",");
const MODES: { mode: GreetingMode; label: TranslationKey }[] = [
  { mode: "unsloth", label: "settings.profile.greetingUnsloth" },
  { mode: "customIcon", label: "settings.profile.greetingCustomIcon" },
  { mode: "textOnly", label: "settings.profile.greetingTextOnly" },
  { mode: "image", label: "settings.profile.greetingImage" },
  { mode: "off", label: "settings.profile.greetingOff" },
];
// "Off" keeps writing showGreetingSloth, so the old switch's value maps straight onto a mode.
const MODE_PATCH = {
  unsloth: {
    showGreetingSloth: true,
    greetingStyle: "text",
    greetingIcon: "sloth",
  },
  customIcon: {
    showGreetingSloth: true,
    greetingStyle: "text",
    greetingIcon: "custom",
  },
  textOnly: {
    showGreetingSloth: true,
    greetingStyle: "text",
    greetingIcon: "none",
  },
  image: { showGreetingSloth: true, greetingStyle: "image" },
  off: { showGreetingSloth: false },
} as const;

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
  if (icon === "custom") {
    return "customIcon";
  }
  return icon === "none" ? "textOnly" : "unsloth";
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
  return { upload, error, setError, busy };
}

// Every greeting option lives in this one row: the mode list, then the upload or text for that mode.
export function GreetingSettingsRows() {
  const t = useT();
  const mode = useGreetingMode();
  const setGreeting = useUserProfileStore((s) => s.setGreeting);
  const text = useUserProfileStore((s) => s.greetingText);
  const iconDataUrl = useUserProfileStore((s) => s.greetingIconDataUrl);
  const imageDataUrl = useUserProfileStore((s) => s.greetingImageDataUrl);
  const { upload, error, setError, busy } = useGreetingUpload();
  const input = useRef<HTMLInputElement>(null);
  const uploadField: UploadField | null =
    mode === "customIcon"
      ? "greetingIconDataUrl"
      : mode === "image"
        ? "greetingImageDataUrl"
        : null;
  const preview =
    mode === "customIcon"
      ? iconDataUrl
      : mode === "image"
        ? imageDataUrl
        : null;
  const showsText =
    mode === "unsloth" || mode === "customIcon" || mode === "textOnly";
  const label = MODES.find((m) => m.mode === mode)?.label ?? MODES[0].label;
  return (
    <SettingsRow
      label={t("settings.profile.greetingSloth")}
      description={t("settings.profile.greetingSlothDescription")}
    >
      {/* Outside the popover so a pick still lands if the file dialog closes it. */}
      <input
        ref={input}
        className="sr-only"
        tabIndex={-1}
        aria-label={`${t("settings.profile.greetingSloth")}: ${t("chatBackground.upload")}`}
        type="file"
        accept={ACCEPT}
        disabled={busy || !uploadField}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (uploadField) {
            upload(uploadField, file);
          }
        }}
      />
      <Popover onOpenChange={(open) => open || setError(null)}>
        <PopoverTrigger asChild={true}>
          <Button
            variant="picker"
            id="profile-greeting-sloth"
            aria-label={t("settings.profile.greetingSloth")}
            className="w-40 justify-between font-normal"
          >
            {t(label)}
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={2}
              className="size-4 text-muted-foreground"
            />
          </Button>
        </PopoverTrigger>
        <PopoverContent
          align="end"
          className="w-72 gap-0 rounded-lg p-1 font-heading"
          data-greeting-menu=""
        >
          <div
            role="radiogroup"
            aria-label={t("settings.profile.greetingSloth")}
          >
            {MODES.map((m) => (
              <button
                key={m.mode}
                type="button"
                role="radio"
                aria-checked={mode === m.mode}
                className="relative flex w-full cursor-pointer items-center rounded-md py-2 pr-9 pl-3 text-left text-sm outline-hidden hover:bg-accent focus-visible:bg-accent"
                onClick={() => {
                  setError(null);
                  setGreeting(MODE_PATCH[m.mode]);
                }}
              >
                {t(m.label)}
                {mode === m.mode && (
                  <HugeiconsIcon
                    icon={MenuTickIcon}
                    strokeWidth={2}
                    className="absolute right-3 size-4"
                  />
                )}
              </button>
            ))}
          </div>
          {(uploadField || showsText) && (
            <>
              <div className="-mx-1 my-1 h-px bg-border" />
              <div className="flex flex-col gap-2 px-3 pt-1 pb-2">
                {uploadField && (
                  <div className="flex items-center gap-2">
                    {preview && (
                      <img
                        src={preview}
                        alt=""
                        className="h-8 max-w-24 shrink-0 object-contain"
                        data-greeting-preview=""
                      />
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
                        : t(
                            preview
                              ? "chatBackground.replace"
                              : "chatBackground.upload",
                          )}
                    </Button>
                  </div>
                )}
                {uploadField && (
                  <p className="text-xs text-muted-foreground">
                    {t("settings.profile.greetingUploadHint")}
                  </p>
                )}
                {error && (
                  <p role="alert" className="text-xs text-destructive">
                    {t(
                      error === "file"
                        ? "settings.profile.greetingFileError"
                        : "settings.profile.greetingImageError",
                    )}
                  </p>
                )}
                {showsText && (
                  <>
                    <p className="text-xs text-muted-foreground">
                      {t("settings.profile.greetingText")}
                    </p>
                    <Input
                      aria-label={t("settings.profile.greetingText")}
                      value={text}
                      maxLength={GREETING_TEXT_MAX_LENGTH}
                      placeholder={t("welcome.onYourMind")}
                      onChange={(event) =>
                        setGreeting({ greetingText: event.target.value })
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      {t("settings.profile.greetingTextDescription")}
                    </p>
                  </>
                )}
              </div>
            </>
          )}
        </PopoverContent>
      </Popover>
    </SettingsRow>
  );
}
