// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Memory only: the raw key is shown once and must not outlive the page.

import { create } from "zustand";

interface FreshApiKeyState {
  key: string | null;
  setKey: (key: string) => void;
  forgetIfRevoked: (keyPrefix: string) => void;
}

// auth/storage.py: key_prefix is the 8 characters after this marker.
const RAW_KEY_MARKER = "sk-unsloth-";

export const useFreshApiKeyStore = create<FreshApiKeyState>((set, get) => ({
  key: null,
  setKey: (key) => set({ key }),
  forgetIfRevoked: (keyPrefix) => {
    const key = get().key;
    if (
      key !== null &&
      keyPrefix &&
      key.startsWith(`${RAW_KEY_MARKER}${keyPrefix}`)
    ) {
      set({ key: null });
    }
  },
}));
