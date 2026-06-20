// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type AvatarShape = "circle" | "rounded";

export interface UserProfileState {
  displayName: string;
  // Preferred name used to address the user (greetings, etc.).
  nickname: string;
  avatarDataUrl: string | null;
  // Avatar outline: full circle or rounded rectangle.
  avatarShape: AvatarShape;
  setDisplayName: (displayName: string) => void;
  setNickname: (nickname: string) => void;
  setAvatarDataUrl: (avatarDataUrl: string | null) => void;
  setAvatarShape: (avatarShape: AvatarShape) => void;
}

export const useUserProfileStore = create<UserProfileState>()(
  persist(
    (set) => ({
      displayName: "",
      nickname: "",
      avatarDataUrl: null,
      avatarShape: "circle",
      setDisplayName: (displayName) => set({ displayName }),
      setNickname: (nickname) => set({ nickname }),
      setAvatarDataUrl: (avatarDataUrl) => set({ avatarDataUrl }),
      setAvatarShape: (avatarShape) => set({ avatarShape }),
    }),
    { name: "unsloth_user_profile" },
  ),
);
