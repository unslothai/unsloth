// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Read only by api-monitor/panel-placement; the notification rail is anchored in CSS instead.

import { create } from "zustand";

export type MonitorFrame = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

/** Identity matters: an exiting monitor unmounts last and must clear only its own box. */
export type MonitorFramePublisher = object;

interface MonitorFrameState {
  /** Not merged: the bounding box of a left monitor and a right composer covers empty space. */
  frames: ReadonlyMap<MonitorFramePublisher, MonitorFrame>;
  setFrame: (publisher: MonitorFramePublisher, frame: MonitorFrame) => void;
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
  // Written on every reconcile, so no-op writes must not notify.
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
