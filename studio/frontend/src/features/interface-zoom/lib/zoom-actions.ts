// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  stepInterfaceScale,
  useInterfaceScaleStore,
} from "@/features/settings";
import { create } from "zustand";
import type { ZoomDirection } from "./zoom-chords.ts";

interface ZoomPopupState {
  /** Bumped on every zoom to restart the hide timer. */
  token: number;
  open: boolean;
  show: () => void;
  hide: () => void;
}

export const useZoomPopupStore = create<ZoomPopupState>()((set) => ({
  token: 0,
  open: false,
  show: () => set((s) => ({ open: true, token: s.token + 1 })),
  hide: () => set({ open: false }),
}));

/** Step the interface scale and show the popup. */
export function zoomInterface(direction: ZoomDirection): void {
  const scale = useInterfaceScaleStore.getState();
  if (direction === 0) scale.reset();
  else scale.setScale(stepInterfaceScale(scale.scale, direction));
  useZoomPopupStore.getState().show();
}

// Guards against the macOS View menu repeating a chord the page already handled.
const MENU_ECHO_MS = 150;
let lastChordZoomAt = Number.NEGATIVE_INFINITY;

export function zoomInterfaceFromChord(direction: ZoomDirection): void {
  lastChordZoomAt = performance.now();
  zoomInterface(direction);
}

export function zoomInterfaceFromMenu(direction: ZoomDirection): void {
  if (performance.now() - lastChordZoomAt < MENU_ECHO_MS) return;
  zoomInterface(direction);
}
