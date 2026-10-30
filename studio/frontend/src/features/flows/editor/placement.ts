// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { RerouteDir } from "../stores/reroute-graph";

/** The Nodes panel's placement tool: the node type armed for click-to-place, if any, and which way
 *  the next free reroute faces (R turns it). */
export const usePlacement = create<{
  armed: string | null;
  rerouteDir: RerouteDir;
}>(() => ({
  armed: null,
  rerouteDir: "right",
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

export const PLACE_GROUP = "flows:group";
export const PLACE_NOTE = "flows:note";

/** Placeables that are not catalog cards: their box and where the pointer holds it, in flow units. */
export const PLACE_SHAPES: Record<
  string,
  { size: [number, number]; grab: [number, number] }
> = {
  reroute: { size: [28, 16], grab: [14, 8] },
  [PLACE_GROUP]: { size: [480, 320], grab: [20, 20] },
  [PLACE_NOTE]: { size: [240, 140], grab: [20, 20] },
};

/** The rect a shape placed at `flow` takes; the snapped point is the grab point, so a reroute centres on a dot. */
export function placedRect(
  type: string,
  flow: { x: number; y: number },
  snap: boolean,
): [number, number, number, number] | null {
  const shape = PLACE_SHAPES[type];
  if (!shape) return null;
  const [x, y] = snapPoint([flow.x, flow.y], snap);
  return [x - shape.grab[0], y - shape.grab[1], ...shape.size];
}

/** Empty canvas, or the body of a group or note: a card lands on top of those. */
export function placeableTarget(target: Element | null): boolean {
  if (!target) return false;
  if (target.classList.contains("react-flow__pane")) return true;
  if (target.closest("button, input, textarea, select, a, [role='button']"))
    return false;
  return !!target.closest(
    ".react-flow__node-flowGroup, .react-flow__node-flowNote",
  );
}
