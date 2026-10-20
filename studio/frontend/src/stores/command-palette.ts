// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { captureFocusedElement } from "@/lib/focus";
import { create } from "zustand";

interface CommandPaletteStore {
  isOpen: boolean;
  // Handed to the dialog an action opens, since the palette unmounts before that dialog closes.
  opener: HTMLElement | null;
  close: () => void;
  toggle: () => void;
  setOpen: (open: boolean) => void;
}

export const useCommandPaletteStore = create<CommandPaletteStore>((set) => ({
  isOpen: false,
  opener: null,
  close: () => set({ isOpen: false }),
  toggle: () =>
    set((s) =>
      s.isOpen
        ? { isOpen: false }
        : { isOpen: true, opener: captureFocusedElement() },
    ),
  setOpen: (isOpen) =>
    set((s) => {
      if (!isOpen) return { isOpen: false };
      return s.isOpen ? s : { isOpen: true, opener: captureFocusedElement() };
    }),
}));
