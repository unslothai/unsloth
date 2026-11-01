// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { useT } from "@/i18n";
import { Upload01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRef, useState } from "react";
import { SettingsRow } from "./settings-row";

type UploadField = "greetingIconDataUrl" | "greetingImageDataUrl";
type UploadError = "file" | "image";

const PROFILE_STORAGE_KEY = "unsloth_user_profile";
const ACCEPT = GREETING_IMAGE_TYPES.join(",");

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

function useGreetingUpload(field: UploadField) {
  const setGreeting = useUserProfileStore((s) => s.setGreeting);
  const [error, setError] = useState<UploadError | null>(null);
  const [busy, setBusy] = useState(false);
  const upload = async (file: File | undefined): Promise<boolean> => {
    if (!file) {
      return false;
    }
    setError(null);
    if (!validGreetingUpload(file)) {
      setError("file");
      return false;
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
      return true;
    } catch {
      setError("image");
      return false;
    } finally {
      setBusy(false);
    }
  };
  return { upload, error, busy };
}

function UploadButton({
  hasValue,
  busy,
  onPick,
  label,
}: {
  hasValue: boolean;
  busy: boolean;
  onPick: (file: File | undefined) => Promise<boolean>;
  label: string;
}) {
  const t = useT();
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={input}
        className="sr-only"
        tabIndex={-1}
        aria-label={label}
        type="file"
        accept={ACCEPT}
        disabled={busy}
        onChange={(event) => {
          onPick(event.target.files?.[0]);
          event.target.value = "";
        }}
      />
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
          : t(hasValue ? "chatBackground.replace" : "chatBackground.upload")}
      </Button>
    </>
  );
}

function UploadErrorText({ error }: { error: UploadError | null }) {
  const t = useT();
  if (!error) {
    return null;
  }
  return (
    <p role="alert" className="text-xs text-destructive">
      {t(
        error === "file"
          ? "settings.profile.greetingFileError"
          : "settings.profile.greetingImageError",
      )}
    </p>
  );
}

function GreetingIconRow() {
  const t = useT();
  const icon = useUserProfileStore((s) => s.greetingIcon);
  const iconDataUrl = useUserProfileStore((s) => s.greetingIconDataUrl);
  const setGreeting = useUserProfileStore((s) => s.setGreeting);
  const { upload, error, busy } = useGreetingUpload("greetingIconDataUrl");
  const custom = icon === "custom";
  return (
    <SettingsRow
      label={t("settings.profile.greetingIcon")}
      description={t("settings.profile.greetingIconDescription")}
      below={<UploadErrorText error={error} />}
      wideControl={true}
    >
      <div className="flex items-center gap-2">
        {custom && iconDataUrl && (
          <img
            src={iconDataUrl}
            alt=""
            className="size-8 shrink-0 object-contain"
            data-greeting-icon-preview=""
          />
        )}
        {custom && (
          <UploadButton
            hasValue={Boolean(iconDataUrl)}
            busy={busy}
            label={t("settings.profile.greetingIcon")}
            onPick={upload}
          />
        )}
        <Select
          value={icon}
          onValueChange={(next) => {
            if (next === "sloth" || next === "custom" || next === "none") {
              setGreeting({ greetingIcon: next });
            }
          }}
        >
          <SelectTrigger
            className="w-32"
            aria-label={t("settings.profile.greetingIcon")}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="sloth">
              {t("settings.profile.greetingIconSloth")}
            </SelectItem>
            <SelectItem value="custom">
              {t("settings.profile.greetingIconCustom")}
            </SelectItem>
            <SelectItem value="none">
              {t("settings.profile.noneLabel")}
            </SelectItem>
          </SelectContent>
        </Select>
      </div>
    </SettingsRow>
  );
}

function GreetingTextRow() {
  const t = useT();
  const text = useUserProfileStore((s) => s.greetingText);
  const setGreeting = useUserProfileStore((s) => s.setGreeting);
  return (
    <SettingsRow
      label={t("settings.profile.greetingText")}
      description={t("settings.profile.greetingTextDescription")}
    >
      <Input
        className="w-60"
        aria-label={t("settings.profile.greetingText")}
        value={text}
        maxLength={GREETING_TEXT_MAX_LENGTH}
        placeholder={t("welcome.onYourMind")}
        onChange={(event) => setGreeting({ greetingText: event.target.value })}
      />
    </SettingsRow>
  );
}

function GreetingImageRow() {
  const t = useT();
  const imageDataUrl = useUserProfileStore((s) => s.greetingImageDataUrl);
  const { upload, error, busy } = useGreetingUpload("greetingImageDataUrl");
  return (
    <SettingsRow
      label={t("settings.profile.greetingImage")}
      description={t("settings.profile.greetingImageDescription")}
      below={<UploadErrorText error={error} />}
      wideControl={true}
    >
      <div className="flex items-center gap-2">
        {imageDataUrl && (
          <img
            src={imageDataUrl}
            alt=""
            className="h-8 max-w-24 shrink-0 object-contain"
            data-greeting-image-preview=""
          />
        )}
        <UploadButton
          hasValue={Boolean(imageDataUrl)}
          busy={busy}
          label={t("settings.profile.greetingImage")}
          onPick={upload}
        />
      </div>
    </SettingsRow>
  );
}

// "Off" keeps writing showGreetingSloth, so the old switch's value maps straight onto this select.
export function GreetingSettingsRows() {
  const t = useT();
  const on = useUserProfileStore((s) => s.showGreetingSloth);
  const style = useUserProfileStore((s) => s.greetingStyle);
  const setGreeting = useUserProfileStore((s) => s.setGreeting);
  return (
    <>
      <SettingsRow
        label={t("settings.profile.greetingSloth")}
        description={t("settings.profile.greetingSlothDescription")}
      >
        <Select
          value={on ? style : "off"}
          onValueChange={(next) => {
            if (next === "off") {
              setGreeting({ showGreetingSloth: false });
            } else if (next === "text" || next === "image") {
              setGreeting({ showGreetingSloth: true, greetingStyle: next });
            }
          }}
        >
          <SelectTrigger
            className="w-40"
            id="profile-greeting-sloth"
            aria-label={t("settings.profile.greetingSloth")}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="text">
              {t("settings.profile.greetingStyleText")}
            </SelectItem>
            <SelectItem value="image">
              {t("settings.profile.greetingStyleImage")}
            </SelectItem>
            <SelectItem value="off">
              {t("settings.profile.greetingOff")}
            </SelectItem>
          </SelectContent>
        </Select>
      </SettingsRow>
      {on && style === "text" && (
        <>
          <GreetingIconRow />
          <GreetingTextRow />
        </>
      )}
      {on && style === "image" && <GreetingImageRow />}
    </>
  );
}
