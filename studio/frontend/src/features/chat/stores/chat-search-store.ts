// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { captureFocusedElement } from "@/lib/focus";
import { create } from "zustand";

interface OpenChatSearchOptions {
  opener?: HTMLElement | null;
}

interface ChatSearchStore {
  isOpen: boolean;
  // Radix cannot recover a trigger that unmounted first (the command palette).
  opener: HTMLElement | null;
  open: (options?: OpenChatSearchOptions) => void;
  close: () => void;
  setOpen: (open: boolean) => void;
}

export const useChatSearchStore = create<ChatSearchStore>((set) => ({
  isOpen: false,
  opener: null,
  open: (options) =>
    set((state) =>
      state.isOpen
        ? state
        : {
            isOpen: true,
            opener:
              options?.opener !== undefined
                ? options.opener
                : captureFocusedElement(),
          },
    ),
  close: () => set({ isOpen: false }),
  // Opener survives close: onCloseAutoFocus reads it after this update.
  setOpen: (isOpen) =>
    set((state) =>
      isOpen
        ? state.isOpen
          ? state
          : { isOpen: true, opener: captureFocusedElement() }
        : { isOpen: false },
    ),
}));
