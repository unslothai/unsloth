// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export interface LightboxSubject {
  kind: "character" | "persona";
  id: string;
  name: string;
  hasImage: boolean;
  rev: number;
  /** A line under the name, such as the creator or the persona's first words. */
  caption?: string;
}

interface LightboxState {
  subject: LightboxSubject | null;
  open: (subject: LightboxSubject) => void;
  close: () => void;
}

export const useCharacterLightbox = create<LightboxState>((set) => ({
  subject: null,
  open: (subject) => set({ subject }),
  close: () => set({ subject: null }),
}));

export function openCharacterLightbox(subject: LightboxSubject): void {
  useCharacterLightbox.getState().open(subject);
}
