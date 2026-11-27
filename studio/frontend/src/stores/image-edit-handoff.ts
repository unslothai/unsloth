// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";

// The Images page may mount later or sit off-route, so the source waits here until taken.
interface ImageEditHandoffState {
  pending: string | null;
  offer: (dataUrl: string) => void;
  take: () => string | null;
}

export const useImageEditHandoffStore = create<ImageEditHandoffState>(
  (set, get) => ({
    pending: null,
    offer: (pending) => set({ pending }),
    take: () => {
      const pending = get().pending;
      if (pending) set({ pending: null });
      return pending;
    },
  }),
);

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    useImageEditHandoffStore.setState({ pending: null });
  });
}
