// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { GreetingIcon, GreetingStyle } from "../greeting";

export type AvatarShape = "circle" | "rounded";
export const PROFILE_TEXT_MAX_LENGTH = 200;

export interface UserProfileState {
  displayName: string;
  nickname: string;
  avatarDataUrl: string | null;
  avatarShape: AvatarShape;
  showGreetingSloth: boolean;
  greetingStyle: GreetingStyle;
  greetingIcon: GreetingIcon;
  greetingIconDataUrl: string | null;
  greetingText: string;
  greetingImageDataUrl: string | null;
  setDisplayName: (displayName: string) => void;
  setNickname: (nickname: string) => void;
  setAvatarDataUrl: (avatarDataUrl: string | null) => void;
  setAvatarShape: (avatarShape: AvatarShape) => void;
  setShowGreetingSloth: (showGreetingSloth: boolean) => void;
  setGreeting: (
    patch: Partial<
      Pick<
        UserProfileState,
        | "showGreetingSloth"
        | "greetingStyle"
        | "greetingIcon"
        | "greetingIconDataUrl"
        | "greetingText"
        | "greetingImageDataUrl"
      >
    >,
  ) => void;
}

export const useUserProfileStore = create<UserProfileState>()(
  persist(
    (set) => ({
      displayName: "",
      nickname: "",
      avatarDataUrl: null,
      avatarShape: "circle",
      showGreetingSloth: true,
      greetingStyle: "text",
      greetingIcon: "sloth",
      greetingIconDataUrl: null,
      greetingText: "",
      greetingImageDataUrl: null,
      setDisplayName: (displayName) => set({ displayName }),
      setNickname: (nickname) => set({ nickname }),
      setAvatarDataUrl: (avatarDataUrl) => set({ avatarDataUrl }),
      setAvatarShape: (avatarShape) => set({ avatarShape }),
      setShowGreetingSloth: (showGreetingSloth) => set({ showGreetingSloth }),
      setGreeting: (patch) => set(patch),
    }),
    { name: "unsloth_user_profile" },
  ),
);
