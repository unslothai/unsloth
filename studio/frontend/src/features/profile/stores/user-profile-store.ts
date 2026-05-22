// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface UserProfileState {
  displayName: string;
  avatarDataUrl: string | null;
  setDisplayName: (displayName: string) => void;
  setAvatarDataUrl: (avatarDataUrl: string | null) => void;
}

export const useUserProfileStore = create<UserProfileState>()(
  persist(
    (set) => ({
      displayName: "",
      avatarDataUrl: null,
      setDisplayName: (displayName) => set({ displayName }),
      setAvatarDataUrl: (avatarDataUrl) => set({ avatarDataUrl }),
    }),
    { name: "unsloth_user_profile" },
  ),
);
