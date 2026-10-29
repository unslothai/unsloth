// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/** The Nodes panel's placement tool: the node type armed for click-to-place, if any. */
export const usePlacement = create<{ armed: string | null }>(() => ({
  armed: null,
}));

/** Clicking the armed item again disarms it; another item switches to that one. */
export function togglePlacement(type: string): void {
  const { armed } = usePlacement.getState();
  usePlacement.setState({ armed: armed === type ? null : type });
}

export function stopPlacement(): void {
  if (usePlacement.getState().armed) usePlacement.setState({ armed: null });
}

/** The snap step, equal to the canvas dot gap so snapped cards sit on the dots. */
export const GRID = 20;
/** Where the pointer holds a new card, from its top-left in flow units: on the header's icon. */
export const GRAB: [number, number] = [27, 27];

export function snapPoint(
  [x, y]: [number, number],
  snap: boolean,
): [number, number] {
  return snap
    ? [Math.round(x / GRID) * GRID, Math.round(y / GRID) * GRID]
    : [Math.round(x), Math.round(y)];
}

/** Where a node placed at `flow` (the pointer, in flow coordinates) goes. */
export function placementPosition(
  flow: { x: number; y: number },
  snap: boolean,
): [number, number] {
  return snapPoint([flow.x - GRAB[0], flow.y - GRAB[1]], snap);
}
