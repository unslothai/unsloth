// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { captureFocusedElement } from "@/lib/focus";
import { create } from "zustand";

interface OpenChatSearchOptions {
  opener?: HTMLElement | null;
}

interface ChatSearchStore {
  isOpen: boolean;
  // Preserve the element focused before a dialog handoff. In particular, the
  // command palette unmounts before this dialog closes, so Radix cannot
  // recover its original trigger on its own.
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
  // Do not clear opener on close: onCloseAutoFocus runs after this state
  // update, and needs the original element to restore focus.
  setOpen: (isOpen) =>
    set((state) =>
      isOpen
        ? state.isOpen
          ? state
          : { isOpen: true, opener: captureFocusedElement() }
        : { isOpen: false },
    ),
}));
