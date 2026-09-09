// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Where the floating panels are, in viewport coordinates. The Live monitor,
// the chat composer and the API monitor panel each publish their box here
// while mounted.
//
// Read only by api-monitor/panel-placement, which opens the API panel clear of
// the others. The notification rail is not a reader: it is anchored in CSS.
// Placing it from these boxes is what moved it out of its corner, since every
// input to that placement (composer height, download rows, release notes)
// changes on its own.

import { create } from "zustand";

export type MonitorFrame = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

/**
 * Whose box this is. Identity matters twice over: reopening the monitor during
 * its exit animation leaves two panels mounted at once and the one on its way
 * out unmounts last, so its cleanup must clear only its own; and the monitor
 * and the composer are published side by side.
 */
export type MonitorFramePublisher = object;

interface MonitorFrameState {
  /** Every published box, kept apart. Deliberately not merged into one: the
   *  rectangle around a left-hand monitor and a bottom-right composer spans
   *  the empty space between them, and reading that as a single obstacle
   *  would have the API panel dodge a corner nothing is in. */
  frames: ReadonlyMap<MonitorFramePublisher, MonitorFrame>;
  setFrame: (publisher: MonitorFramePublisher, frame: MonitorFrame) => void;
  /** Drops only this publisher's box; the others still count. */
  clearFrame: (publisher: MonitorFramePublisher) => void;
}

function sameFrame(a: MonitorFrame | null, b: MonitorFrame | null): boolean {
  if (a === null || b === null) return a === b;
  return (
    a.left === b.left &&
    a.top === b.top &&
    a.right === b.right &&
    a.bottom === b.bottom
  );
}

export const useMonitorFrameStore = create<MonitorFrameState>((set) => ({
  frames: new Map(),
  // Written from a layout effect on every reconcile, so no-op writes must not
  // notify: the API monitor panel re-places itself on this.
  setFrame: (publisher, frame) =>
    set((state) => {
      if (sameFrame(state.frames.get(publisher) ?? null, frame)) return state;
      return { frames: new Map(state.frames).set(publisher, frame) };
    }),
  clearFrame: (publisher) =>
    set((state) => {
      if (!state.frames.has(publisher)) return state;
      const frames = new Map(state.frames);
      frames.delete(publisher);
      return { frames };
    }),
}));
