// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which floating panel is in front.
 *
 * The Live resource monitor and the API monitor overlay are both windows: they
 * float over the page, they are dragged, resized and closed, and they both
 * default to the bottom-right corner. They keep out of each other's way
 * geometrically (see api-monitor/panel-placement), and that is the whole
 * answer while there is anywhere to move to. There is not always: a monitor
 * resized to fill the viewport leaves no clear space at all, and then the only
 * thing deciding whether the panel underneath can be reached is z-index.
 *
 * So they share one layer and the one the user touched last comes forward,
 * which is what every window manager does and what the rest of Unsloth's
 * corner overlays already imply -- the notification stack yields to whichever
 * panel is published under it rather than fighting for a number.
 *
 * Deliberately not a stack of arbitrary depth: there are two panels, so "which
 * one is on top" is one value, and a list would only invite a third caller to
 * assume ordering it does not have.
 */

import { create } from "zustand";
import { Z_LAYER } from "./z-layers.ts";

export type FloatingPanelId = "resource-monitor" | "api-monitor";

interface FloatingPanelOrderState {
  /** The panel in front, or null before either has been opened. */
  top: FloatingPanelId | null;
  /** Bring `id` forward. A no-op write must not notify: this runs on pointerdown. */
  raise: (id: FloatingPanelId) => void;
}

export const useFloatingPanelOrderStore = create<FloatingPanelOrderState>(
  (set) => ({
    top: null,
    raise: (id) => set((state) => (state.top === id ? state : { top: id })),
  }),
);

/**
 * The z-index `id` should paint at.
 *
 * Both panels sit on FLOATING_PANEL; the front one takes the single step above
 * it. Nothing is ever two steps up, so the pair can never straddle the layer
 * above them.
 *
 * `hidden` is the one override: a panel with no pixel left showing cannot be
 * clicked, so it cannot be brought back to the front by the same rule as
 * everything else. It comes forward on its own. Only the API monitor panel asks
 * this, because only it opens and places itself; the resource monitor is where
 * the user put it, and whatever is over it is there because they dragged it.
 */
export function floatingPanelZIndex(
  id: FloatingPanelId,
  top: FloatingPanelId | null,
  hidden = false,
): number {
  return hidden || top === id
    ? Z_LAYER.FLOATING_PANEL_TOP
    : Z_LAYER.FLOATING_PANEL;
}

/** `floatingPanelZIndex` for the calling panel, re-read as the front one changes. */
export function useFloatingPanelZIndex(
  id: FloatingPanelId,
  hidden = false,
): number {
  const top = useFloatingPanelOrderStore((state) => state.top);
  return floatingPanelZIndex(id, top, hidden);
}
