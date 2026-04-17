// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getAuthToken } from "@/features/auth";
import { toastError, toastSuccess } from "@/shared/toast";
import { Camera } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { decodeJwtSubject } from "../utils/jwt-subject";
import { resizeImageFileToDataUrl } from "../utils/resize-image-file";
import { useUserProfileStore } from "../stores/user-profile-store";
import { UserAvatar } from "./user-avatar";

const PROFILE_STORAGE_KEY = "unsloth_user_profile";

function readPersistedProfile(): { displayName: string; avatarDataUrl: string | null } | null {
  try {
    const raw = window.localStorage.getItem(PROFILE_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") return null;

    // Zustand persist shape: { state: {...}, version }
    const maybeState = "state" in parsed ? (parsed as { state?: unknown }).state : parsed;
    if (!maybeState || typeof maybeState !== "object") return null;
    const state = maybeState as { displayName?: unknown; avatarDataUrl?: unknown };

    return {
      displayName: typeof state.displayName === "string" ? state.displayName : "",
      avatarDataUrl: typeof state.avatarDataUrl === "string" ? state.avatarDataUrl : null,
    };
  } catch {
    return null;
  }
}

export function ProfilePersonalizationPanel() {
  const displayName = useUserProfileStore((s) => s.displayName);
  const avatarDataUrl = useUserProfileStore((s) => s.avatarDataUrl);
  const setDisplayName = useUserProfileStore((s) => s.setDisplayName);
  const setAvatarDataUrl = useUserProfileStore((s) => s.setAvatarDataUrl);

  const [imageError, setImageError] = useState<string | null>(null);
  const [draftName, setDraftName] = useState(displayName);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const sessionSub = decodeJwtSubject(getAuthToken()) ?? "";
  const previewName = draftName.trim() || sessionSub || "Unsloth";
  const hasNameChanges = useMemo(
    () => draftName.trim() !== displayName.trim(),
    [draftName, displayName],
  );

  const saveName = () => {
    const trimmed = draftName.trim();
    if (trimmed !== draftName) setDraftName(trimmed);
    if (trimmed !== displayName) {
      setDisplayName(trimmed);
      const persisted = readPersistedProfile();
      if (persisted && persisted.displayName === trimmed) {
        toastSuccess("Profile name saved");
      } else {
        toastError(
          "Could not persist profile name",
          "Name updated for this session, but may not persist after reload.",
        );
      }
    }
  };

  const onPickFile = async (file: File | undefined) => {
    if (!file) return;
    setImageError(null);
    try {
      const dataUrl = await resizeImageFileToDataUrl(file);
      setAvatarDataUrl(dataUrl);
      const persisted = readPersistedProfile();
      if (persisted && persisted.avatarDataUrl === dataUrl) {
        toastSuccess("Profile photo updated");
      } else {
        toastError(
          "Could not persist profile photo",
          "Photo updated for this session, but may not persist after reload.",
        );
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : "Could not use this image.";
      setImageError(message);
      toastError("Could not update profile photo", message);
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-[640px] flex-col items-center gap-6 rounded-2xl border border-border/70 bg-muted/10 px-8 py-7">
      <div className="relative">
        <UserAvatar
          name={previewName}
          imageUrl={avatarDataUrl}
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
          className="absolute right-0 bottom-0 -translate-x-[15.625%] -translate-y-[15.625%] flex size-8 items-center justify-center rounded-full border border-border bg-background text-foreground shadow-sm transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          aria-label="Change profile picture"
        >
          <Camera className="size-3.5" strokeWidth={2} />
        </button>
      </div>

      <div className="flex w-full max-w-[560px] flex-col gap-2">
        <Label htmlFor="profile-display-name" className="text-xs font-medium text-muted-foreground">
          Display name
        </Label>
        <div className="flex items-center gap-2">
          <Input
            id="profile-display-name"
            type="text"
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                saveName();
              }
            }}
            autoComplete="off"
            placeholder={sessionSub || "Unsloth"}
            className="h-10 min-w-0 flex-1 rounded-lg text-sm"
          />
          <Button type="button" size="sm" className="h-10 px-5" onClick={saveName} disabled={!hasNameChanges}>
            Save
          </Button>
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
