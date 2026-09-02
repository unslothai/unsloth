// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

import { ACCOUNT_CHANGED_EVENT } from "../../../lib/account-transition.ts";

export type AvatarShape = "circle" | "rounded";
export const PROFILE_TEXT_MAX_LENGTH = 200;

export interface UserProfileState {
  displayName: string;
  nickname: string;
  avatarDataUrl: string | null;
  avatarShape: AvatarShape;
  showGreetingSloth: boolean;
  setDisplayName: (displayName: string) => void;
  setNickname: (nickname: string) => void;
  setAvatarDataUrl: (avatarDataUrl: string | null) => void;
  setAvatarShape: (avatarShape: AvatarShape) => void;
  setShowGreetingSloth: (showGreetingSloth: boolean) => void;
}

export const useUserProfileStore = create<UserProfileState>()(
  persist(
    (set) => ({
      displayName: "",
      nickname: "",
      avatarDataUrl: null,
      avatarShape: "circle",
      showGreetingSloth: true,
      setDisplayName: (displayName) => set({ displayName }),
      setNickname: (nickname) => set({ nickname }),
      setAvatarDataUrl: (avatarDataUrl) => set({ avatarDataUrl }),
      setAvatarShape: (avatarShape) => set({ avatarShape }),
      setShowGreetingSloth: (showGreetingSloth) => set({ showGreetingSloth }),
    }),
    { name: "unsloth_user_profile" },
  ),
);

// The persisted key is cleared when a different account signs in, but this store
// is already hydrated and a same-window removeItem fires no storage event, so
// without this the personalization sync still read the previous account's name,
// nickname and avatar and saved them into the new account's record as migration
// input.
if (typeof window !== "undefined") {
  window.addEventListener(ACCOUNT_CHANGED_EVENT, () => {
    useUserProfileStore.setState({
      displayName: "",
      nickname: "",
      avatarDataUrl: null,
      avatarShape: "circle",
      showGreetingSloth: true,
    });
  });
}
