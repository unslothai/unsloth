// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Wheel travel per step: one mouse notch, or a short touchpad pinch. */
export const WHEEL_ZOOM_STEP_PX = 50;

/** A pause this long starts a new gesture. */
export const WHEEL_ZOOM_IDLE_MS = 250;

const LINE_PX = 40;
const PAGE_PX = 800;

type ZoomWheelEvent = Pick<
  WheelEvent,
  "deltaY" | "deltaMode" | "ctrlKey" | "metaKey" | "altKey" | "timeStamp"
>;

/** Ctrl+wheel to zoom steps (1 in, -1 out). `zoom` converts CSS px deltas to screen px. */
export function createWheelZoomAccumulator(): (
  event: ZoomWheelEvent,
  zoom: number,
) => 1 | -1 | null {
  let travel = 0;
  let lastAt = Number.NEGATIVE_INFINITY;
  return (event, zoom) => {
    if (!event.ctrlKey || event.metaKey || event.altKey || event.deltaY === 0)
      return null;
    const unit =
      event.deltaMode === 1 ? LINE_PX : event.deltaMode === 2 ? PAGE_PX : 1;
    const delta = event.deltaY * unit * zoom;
    // Reset on a pause or a direction change.
    if (
      event.timeStamp - lastAt > WHEEL_ZOOM_IDLE_MS ||
      Math.sign(delta) !== Math.sign(travel)
    ) {
      travel = 0;
    }
    lastAt = event.timeStamp;
    travel += delta;
    if (Math.abs(travel) < WHEEL_ZOOM_STEP_PX) return null;
    const direction = travel < 0 ? 1 : -1;
    // One step per crossing, however large the delta.
    travel = 0;
    return direction;
  };
}
