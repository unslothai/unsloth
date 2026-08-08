// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { captureFocusedElement } from "@/lib/focus";
import { create } from "zustand";

interface CommandPaletteStore {
  isOpen: boolean;
  // Element focused before the palette opened. Actions that open another
  // dialog (Settings) pass this along so closing that dialog can restore
  // focus past the palette, which unmounts in between.
  opener: HTMLElement | null;
  open: () => void;
  close: () => void;
  toggle: () => void;
  setOpen: (open: boolean) => void;
}

export const useCommandPaletteStore = create<CommandPaletteStore>((set) => ({
  isOpen: false,
  opener: null,
  open: () =>
    set((s) =>
      s.isOpen ? s : { isOpen: true, opener: captureFocusedElement() },
    ),
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
