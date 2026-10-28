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

const GRID = 16;
// A new card's top-left sits this far up-left of the pointer, so the pointer lands on its header.
const GRAB: [number, number] = [130, 20];

/** Where a node placed at `flow` (the pointer, in flow coordinates) goes. */
export function placementPosition(
  flow: { x: number; y: number },
  snap: boolean,
): [number, number] {
  const x = flow.x - GRAB[0];
  const y = flow.y - GRAB[1];
  return snap
    ? [Math.round(x / GRID) * GRID, Math.round(y / GRID) * GRID]
    : [Math.round(x), Math.round(y)];
}
