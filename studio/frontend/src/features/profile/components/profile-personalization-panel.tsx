


import { publicAssetUrl } from "@/components/mascot-img";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { getAuthToken } from "@/features/auth";
import { cn } from "@/lib/utils";
import { useT } from "@/i18n";
import { toastError, toastSuccess } from "@/shared/toast";
import {
  Delete02Icon,
  Edit03Icon,
  Image01Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { SLOTH_AVATARS } from "../sloth-avatars";
import { decodeJwtSubject } from "../utils/jwt-subject";
import { resizeImageFileToDataUrl } from "../utils/resize-image-file";
import {
  PROFILE_TEXT_MAX_LENGTH,
  useUserProfileStore,
} from "../stores/user-profile-store";
import { UserAvatar } from "./user-avatar";

const PROFILE_STORAGE_KEY = "unsloth_user_profile";
const SLOTH_NAME = /^large\s+/i;
const PNG_SUFFIX = /\.png$/i;

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
    const maybeState =
      "state" in parsed ? (parsed as { state?: unknown }).state : parsed;
    if (!maybeState || typeof maybeState !== "object") return null;
    const state = maybeState as {
      displayName?: unknown;
      nickname?: unknown;
      avatarDataUrl?: unknown;
    };

    return {
      displayName:
        typeof state.displayName === "string" ? state.displayName : "",
      nickname: typeof state.nickname === "string" ? state.nickname : "",
      avatarDataUrl:
        typeof state.avatarDataUrl === "string" ? state.avatarDataUrl : null,
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

  const [imageError, setImageError] = useState<string | null>(null);
  const [draftName, setDraftName] = useState(displayName);
  const [draftNickname, setDraftNickname] = useState(nickname);
  const [pickerOpen, setPickerOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastDisplayNameRef = useRef(displayName);
  const lastNicknameRef = useRef(nickname);

  const sessionSub = decodeJwtSubject(getAuthToken()) ?? "";
  const previewName = draftName.trim() || sessionSub || "Unsloth";

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

  // Committed on blur and on Enter rather than behind a Save button, so each
  // field is a single row like the rest of Settings.
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

  // Escape, or any programmatic close, unmounts the tab without dispatching a
  // blur, which would drop whatever was typed. Commit the drafts on the way
  // out; both saves no-op on an unchanged value, so a double commit is safe.
  const flushDrafts = useRef<() => void>(() => {});
  useEffect(() => {
    flushDrafts.current = () => {
      saveName();
      saveNickname();
    };
  });
  useEffect(() => () => flushDrafts.current(), []);

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

  return (
    <div className="flex w-full flex-col">
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

      <div className="flex items-center gap-10 py-6 pr-2">
        <div className="relative shrink-0">
          {/* The picture itself is the shortcut to "upload a photo"; the pencil
              opens the rest of the options. */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            aria-label={t("settings.profile.changePicture")}
            className="group relative block rounded-full focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <UserAvatar
              name={previewName}
              imageUrl={shownAvatar}
              size="lg"
              className="size-[128px] text-[calc(3.2rem*var(--ui-font-scale,1))]"
            />
            <span className="absolute inset-0 flex items-center justify-center rounded-full bg-black/45 opacity-0 transition-opacity group-hover:opacity-100">
              <HugeiconsIcon
                icon={Image01Icon}
                className="size-8 text-white"
                strokeWidth={2}
              />
            </span>
          </button>

          <Popover open={pickerOpen} onOpenChange={setPickerOpen}>
            <PopoverTrigger asChild={true}>
              <button
                type="button"
                aria-label={t("settings.profile.pictureOptions")}
                title={t("settings.profile.pictureOptions")}
                className="absolute top-[85.36%] left-[85.36%] flex size-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-border bg-background text-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:border-transparent dark:bg-white/[0.14] dark:hover:bg-white/20"
              >
                <HugeiconsIcon
                  icon={Edit03Icon}
                  className="size-4.5"
                  strokeWidth={2}
                />
              </button>
            </PopoverTrigger>
            <PopoverContent
              align="start"
              sideOffset={10}
              className="w-[320px] gap-4 p-4"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="text-ui-11 font-medium uppercase tracking-wide text-muted-foreground">
                  {t("settings.profile.avatarShape")}
                </span>
                <div className="hub-tab-toggle flex h-8 shrink-0 items-center rounded-full">
                  {(["circle", "rounded"] as const).map((shape) => (
                    <button
                      key={shape}
                      type="button"
                      onClick={() => setAvatarShape(shape)}
                      aria-pressed={avatarShape === shape}
                      className={cn(
                        "inline-flex h-8 items-center justify-center rounded-full px-3.5 text-ui-13 font-medium transition-colors",
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

              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  className="h-9 w-fit gap-2 rounded-full px-4 text-sm"
                >
                  <HugeiconsIcon
                    icon={Upload01Icon}
                    className="size-4"
                    strokeWidth={2}
                  />
                  {t("settings.profile.uploadPhoto")}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => pickAvatarValue(null)}
                  disabled={shownAvatar === null}
                  aria-label={t("settings.profile.removePhoto")}
                  title={t("settings.profile.removePhoto")}
                  className="size-9 shrink-0 rounded-full p-0 text-muted-foreground"
                >
                  <HugeiconsIcon
                    icon={Delete02Icon}
                    className="size-4"
                    strokeWidth={2}
                  />
                </Button>
              </div>

              <div className="flex flex-col gap-2">
                <span className="text-ui-11 font-medium uppercase tracking-wide text-muted-foreground">
                  {t("settings.profile.chooseSloth")}
                </span>
                <div className="grid grid-cols-7 gap-2">
                  {SLOTH_AVATARS.map((path) => {
                    const url = publicAssetUrl(path);
                    const selected = shownAvatar === url;
                    const label =
                      path
                        .split("/")
                        .pop()
                        ?.replace(PNG_SUFFIX, "")
                        .replace(SLOTH_NAME, "")
                        .trim() ?? "sloth";
                    return (
                      <button
                        key={path}
                        type="button"
                        onClick={() => pickAvatarValue(url)}
                        aria-pressed={selected}
                        aria-label={label}
                        title={label}
                        className={cn(
                          // No transition here: animating the ring makes the old
                          // icon's selection border linger when switching sloths.
                          "relative aspect-square overflow-hidden rounded-full bg-muted ring-1 ring-border hover:ring-ring focus-visible:outline-none focus-visible:ring-ring",
                          selected &&
                            "ring-2 ring-ring-strong hover:ring-ring-strong",
                        )}
                      >
                        <img
                          src={url}
                          alt=""
                          loading="lazy"
                          className="size-full object-cover"
                        />
                      </button>
                    );
                  })}
                </div>
              </div>
            </PopoverContent>
          </Popover>
        </div>

        {/* Name fields sit beside the picture. These are not SettingsRows, so
            data-settings-label is set by hand for settings search. */}
        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <div
            data-settings-label={t("settings.profile.displayName")}
            className="flex min-w-0 flex-col gap-1.5"
          >
            <Label
              htmlFor="profile-display-name"
              className="text-xs font-medium text-muted-foreground"
            >
              {t("settings.profile.displayName")}
            </Label>
            <Input
              id="profile-display-name"
              type="text"
              value={draftName}
              maxLength={PROFILE_TEXT_MAX_LENGTH}
              onChange={(e) => setDraftName(e.target.value)}
              onBlur={saveName}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.currentTarget.blur();
                }
              }}
              autoComplete="off"
              placeholder={sessionSub || "Unsloth"}
              className="h-9 w-full rounded-full text-sm"
            />
          </div>

          <div
            data-settings-label={t("settings.profile.nickname")}
            className="flex min-w-0 flex-col gap-1.5"
          >
            <Label
              htmlFor="profile-nickname"
              className="text-xs font-medium text-muted-foreground"
            >
              {t("settings.profile.nickname")}
            </Label>
            <Input
              id="profile-nickname"
              type="text"
              value={draftNickname}
              maxLength={PROFILE_TEXT_MAX_LENGTH}
              onChange={(e) => setDraftNickname(e.target.value)}
              onBlur={saveNickname}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.currentTarget.blur();
                }
              }}
              autoComplete="off"
              placeholder={t("settings.profile.nicknamePlaceholder")}
              className="h-9 w-full rounded-full text-sm"
            />
          </div>
        </div>
      </div>

      {imageError ? (
        <p className="pt-2 text-xs text-destructive" role="alert">
          {imageError}
        </p>
      ) : null}
    </div>
  );
}
