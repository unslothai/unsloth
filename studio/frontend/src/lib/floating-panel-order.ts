// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Live and API monitors share one layer and the last-touched comes forward, since a full-viewport
 * monitor leaves no clear space. Two panels, so "which is on top" is one value, not a stack.
 */

import { create } from "zustand";
import { Z_LAYER } from "./z-layers.ts";

export type FloatingPanelId = "resource-monitor" | "api-monitor";

interface FloatingPanelOrderState {
  top: FloatingPanelId | null;
  /** Runs on pointerdown, so a no-op write must not notify. */
  raise: (id: FloatingPanelId) => void;
}

export const useFloatingPanelOrderStore = create<FloatingPanelOrderState>(
  (set) => ({
    top: null,
    raise: (id) => set((state) => (state.top === id ? state : { top: id })),
  }),
);

/** The front panel takes one step above FLOATING_PANEL, never two. `hidden` (API monitor only)
 * comes forward on its own, since a fully covered panel cannot be clicked. */
export function floatingPanelZIndex(
  id: FloatingPanelId,
  top: FloatingPanelId | null,
  hidden = false,
): number {
  return hidden || top === id
    ? Z_LAYER.FLOATING_PANEL_TOP
    : Z_LAYER.FLOATING_PANEL;
}

export function useFloatingPanelZIndex(
  id: FloatingPanelId,
  hidden = false,
): number {
  const top = useFloatingPanelOrderStore((state) => state.top);
  return floatingPanelZIndex(id, top, hidden);
}
