// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";

// "Edit image" from the Library: the Images page may mount on navigation or already be mounted
// off-route, so the source image waits here until the page takes it.
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
