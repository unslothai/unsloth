// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { publicAssetUrl } from "@/components/mascot-img";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { getAuthToken } from "@/features/auth";
import { cn } from "@/lib/utils";
import { useT } from "@/i18n";
import { toastError, toastSuccess } from "@/shared/toast";
import { Edit03Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { SLOTH_AVATARS } from "../sloth-avatars";
import { decodeJwtSubject } from "../utils/jwt-subject";
import { resizeImageFileToDataUrl } from "../utils/resize-image-file";
import {
  PROFILE_TEXT_MAX_LENGTH,
  useUserProfileStore,
} from "../stores/user-profile-store";
import { UserAvatar } from "./user-avatar";

const PROFILE_STORAGE_KEY = "unsloth_user_profile";

function readPersistedProfile(): {
  displayName: string;
  nickname: string;
  avatarDataUrl: string | null;
} | null {
  try {
    const raw = window.localStorage.getItem(PROFILE_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") return null;

    // Zustand persist shape: { state: {...}, version }
    const maybeState = "state" in parsed ? (parsed as { state?: unknown }).state : parsed;
    if (!maybeState || typeof maybeState !== "object") return null;
    const state = maybeState as {
      displayName?: unknown;
      nickname?: unknown;
      avatarDataUrl?: unknown;
    };

    return {
      displayName: typeof state.displayName === "string" ? state.displayName : "",
      nickname: typeof state.nickname === "string" ? state.nickname : "",
      avatarDataUrl: typeof state.avatarDataUrl === "string" ? state.avatarDataUrl : null,
    };
  } catch {
    return null;
  }
}

export function ProfilePersonalizationPanel() {
  const t = useT();
  const displayName = useUserProfileStore((s) => s.displayName);
  const nickname = useUserProfileStore((s) => s.nickname);
  const avatarDataUrl = useUserProfileStore((s) => s.avatarDataUrl);
  const setDisplayName = useUserProfileStore((s) => s.setDisplayName);
  const setNickname = useUserProfileStore((s) => s.setNickname);
  const setAvatarDataUrl = useUserProfileStore((s) => s.setAvatarDataUrl);
  const avatarShape = useUserProfileStore((s) => s.avatarShape);
  const setAvatarShape = useUserProfileStore((s) => s.setAvatarShape);
  const showGreetingSloth = useUserProfileStore((s) => s.showGreetingSloth);
  const setShowGreetingSloth = useUserProfileStore(
    (s) => s.setShowGreetingSloth,
  );

  const [imageError, setImageError] = useState<string | null>(null);
  const [draftName, setDraftName] = useState(displayName);
  const [draftNickname, setDraftNickname] = useState(nickname);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastDisplayNameRef = useRef(displayName);
  const lastNicknameRef = useRef(nickname);

  const sessionSub = decodeJwtSubject(getAuthToken()) ?? "";
  const previewName = draftName.trim() || sessionSub || "Unsloth";
  const hasNameChanges = useMemo(
    () => draftName.trim() !== displayName.trim(),
    [draftName, displayName],
  );
  const hasNicknameChanges = useMemo(
    () => draftNickname.trim() !== nickname.trim(),
    [draftNickname, nickname],
  );

  useEffect(() => {
    const previous = lastDisplayNameRef.current;
    lastDisplayNameRef.current = displayName;
    setDraftName((draft) => (draft === previous ? displayName : draft));
  }, [displayName]);

  useEffect(() => {
    const previous = lastNicknameRef.current;
    lastNicknameRef.current = nickname;
    setDraftNickname((draft) => (draft === previous ? nickname : draft));
  }, [nickname]);

  const saveName = () => {
    const trimmed = draftName.trim();
    if (trimmed !== draftName) setDraftName(trimmed);
    if (trimmed !== displayName) {
      setDisplayName(trimmed);
      const persisted = readPersistedProfile();
      if (persisted && persisted.displayName === trimmed) {
        toastSuccess(t("settings.profile.nameSaved"));
      } else {
        toastError(
          t("settings.profile.namePersistErrorTitle"),
          t("settings.profile.namePersistErrorDescription"),
        );
      }
    }
  };

  const saveNickname = () => {
    const trimmed = draftNickname.trim();
    if (trimmed !== draftNickname) setDraftNickname(trimmed);
    if (trimmed !== nickname) {
      setNickname(trimmed);
      const persisted = readPersistedProfile();
      if (persisted && persisted.nickname === trimmed) {
        toastSuccess(t("settings.profile.nicknameSaved"));
      } else {
        toastError(
          t("settings.profile.namePersistErrorTitle"),
          t("settings.profile.namePersistErrorDescription"),
        );
      }
    }
  };

  const applyAvatar = (value: string | null) => {
    setAvatarDataUrl(value);
    const persisted = readPersistedProfile();
    if (persisted && persisted.avatarDataUrl === value) {
      toastSuccess(t("settings.profile.photoUpdated"));
    } else {
      toastError(
        t("settings.profile.photoPersistErrorTitle"),
        t("settings.profile.photoPersistErrorDescription"),
      );
    }
  };

  const onPickFile = async (file: File | undefined) => {
    if (!file) return;
    setImageError(null);
    try {
      applyAvatar(await resizeImageFileToDataUrl(file));
    } catch (e) {
      const message =
        e instanceof Error ? e.message : t("settings.profile.imageUseError");
      setImageError(message);
      toastError(t("settings.profile.photoUpdateErrorTitle"), message);
    }
  };

  // The avatar is shown all over the app (sidebar, chat messages, greeting),
  // so writing it to the store can trigger a wide re-render. Mark the picked
  // value locally first so its ring moves this frame, then commit the store
  // write on the next frame. Boxed because null is a valid pick (no picture).
  const [pendingAvatar, setPendingAvatar] = useState<{
    value: string | null;
  } | null>(null);
  const shownAvatar = pendingAvatar ? pendingAvatar.value : avatarDataUrl;

  useEffect(() => {
    if (pendingAvatar && avatarDataUrl === pendingAvatar.value) {
      setPendingAvatar(null);
    }
  }, [avatarDataUrl, pendingAvatar]);

  const pickAvatarValue = (value: string | null) => {
    setImageError(null);
    setPendingAvatar({ value });
    requestAnimationFrame(() => applyAvatar(value));
  };

  const pickSloth = (path: string) => {
    pickAvatarValue(publicAssetUrl(path));
  };

  return (
    <div className="mx-auto flex w-full max-w-[640px] flex-col items-center gap-6 rounded-2xl border border-border/70 bg-muted/10 px-8 py-7">
      <div className="relative">
        <UserAvatar
          name={previewName}
          imageUrl={shownAvatar}
          size="lg"
          className="size-[124px] text-[3.15rem]"
        />
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/gif"
          className="sr-only"
          onChange={(e) => {
            void onPickFile(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="absolute right-0 bottom-0 -translate-x-[15.625%] -translate-y-[15.625%] flex size-8 items-center justify-center rounded-full border border-border bg-background text-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          aria-label={t("settings.profile.changePicture")}
        >
          <HugeiconsIcon icon={Edit03Icon} className="size-3.5" strokeWidth={2} />
        </button>
      </div>

      <div
        data-settings-label={t("settings.profile.displayName")}
        className="flex w-full max-w-[560px] flex-col gap-2"
      >
        <Label htmlFor="profile-display-name" className="text-xs font-medium text-muted-foreground">
          {t("settings.profile.displayName")}
        </Label>
        <div className="flex items-center gap-2">
          <Input
            id="profile-display-name"
            type="text"
            value={draftName}
            maxLength={PROFILE_TEXT_MAX_LENGTH}
            onChange={(e) => setDraftName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                saveName();
              }
            }}
            autoComplete="off"
            placeholder={sessionSub || "Unsloth"}
            className="h-10 min-w-0 flex-1 rounded-full text-sm"
          />
          <Button type="button" size="sm" className="h-10 px-5" onClick={saveName} disabled={!hasNameChanges}>
            {t("common.save")}
          </Button>
        </div>
      </div>

      <div
        data-settings-label={t("settings.profile.nickname")}
        className="flex w-full max-w-[560px] flex-col gap-2"
      >
        <Label htmlFor="profile-nickname" className="text-xs font-medium text-muted-foreground">
          {t("settings.profile.nickname")}
        </Label>
        <div className="flex items-center gap-2">
          <Input
            id="profile-nickname"
            type="text"
            value={draftNickname}
            maxLength={PROFILE_TEXT_MAX_LENGTH}
            onChange={(e) => setDraftNickname(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                saveNickname();
              }
            }}
            autoComplete="off"
            placeholder={t("settings.profile.nicknamePlaceholder")}
            className="h-10 min-w-0 flex-1 rounded-full text-sm"
          />
          <Button type="button" size="sm" className="h-10 px-5" onClick={saveNickname} disabled={!hasNicknameChanges}>
            {t("common.save")}
          </Button>
        </div>
      </div>

      <div
        data-settings-label={t("settings.profile.avatarShape")}
        className="flex w-full max-w-[560px] flex-col gap-2"
      >
        <Label className="text-xs font-medium text-muted-foreground">
          {t("settings.profile.avatarShape")}
        </Label>
        <div className="hub-tab-toggle inline-flex h-8 w-fit items-center rounded-full">
          {(["circle", "rounded"] as const).map((shape) => (
            <button
              key={shape}
              type="button"
              onClick={() => setAvatarShape(shape)}
              aria-pressed={avatarShape === shape}
              className={cn(
                "inline-flex h-8 items-center rounded-full px-4 text-[13px] font-medium transition-colors",
                avatarShape === shape
                  ? "hub-tab-toggle-pill text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {shape === "circle"
                ? t("settings.profile.avatarShapeCircle")
                : t("settings.profile.avatarShapeRounded")}
            </button>
          ))}
        </div>
      </div>

      <div
        data-settings-label={t("settings.profile.greetingSloth")}
        className="flex w-full max-w-[560px] items-center justify-between gap-4"
      >
        <div className="flex min-w-0 flex-col gap-0.5">
          <Label
            htmlFor="profile-greeting-sloth"
            className="text-xs font-medium text-muted-foreground"
          >
            {t("settings.profile.greetingSloth")}
          </Label>
          <p className="text-xs text-muted-foreground/75">
            {t("settings.profile.greetingSlothDescription")}
          </p>
        </div>
        <Switch
          id="profile-greeting-sloth"
          checked={showGreetingSloth}
          onCheckedChange={setShowGreetingSloth}
        />
      </div>

      <div className="flex w-full max-w-[560px] flex-col gap-2">
        <Label className="text-xs font-medium text-muted-foreground">
          {t("settings.profile.chooseSloth")}
        </Label>
        <div className="grid grid-cols-7 gap-2 sm:grid-cols-9">
          {SLOTH_AVATARS.map((path) => {
            const url = publicAssetUrl(path);
            const selected = shownAvatar === url;
            const label =
              path.split("/").pop()?.replace(/\.png$/i, "").replace(/^large\s+/i, "").trim() ??
              "sloth";
            return (
              <button
                key={path}
                type="button"
                onClick={() => pickSloth(path)}
                aria-pressed={selected}
                aria-label={label}
                title={label}
                className={cn(
                  // No transition here: animating the ring makes the old
                  // icon's selection border linger when switching sloths.
                  "relative aspect-square overflow-hidden rounded-full bg-muted ring-1 ring-border hover:ring-ring focus-visible:outline-none focus-visible:ring-ring",
                  // Selection keeps the 1px weight, only darker.
                  selected && "ring-ring-strong hover:ring-ring-strong",
                )}
              >
                <img src={url} alt="" loading="lazy" className="size-full object-cover" />
              </button>
            );
          })}
          <button
            type="button"
            onClick={() => pickAvatarValue(null)}
            aria-pressed={shownAvatar === null}
            aria-label={t("settings.profile.noPicture")}
            title={t("settings.profile.noPicture")}
            className={cn(
              "relative flex aspect-square items-center justify-center overflow-hidden rounded-full bg-muted text-muted-foreground ring-1 ring-border hover:ring-ring focus-visible:outline-none focus-visible:ring-ring",
              shownAvatar === null && "ring-ring-strong hover:ring-ring-strong",
            )}
          >
            <span className="text-[11px] font-medium">
              {t("settings.profile.noneLabel")}
            </span>
          </button>
        </div>
      </div>

      {imageError ? (
        <p className="w-full text-xs text-destructive" role="alert">
          {imageError}
        </p>
      ) : null}
    </div>
  );
}
