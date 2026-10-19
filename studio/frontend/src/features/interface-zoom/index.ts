// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { InterfaceZoom } from "./components/interface-zoom.tsx";
export {
  type ZoomScope,
  registerZoomScope,
  zoomInterfaceFromMenu,
  zoomScopeFor,
  zoomScopeFromChord,
} from "./lib/zoom-actions.ts";
export { type ZoomDirection, zoomDirectionForKey } from "./lib/zoom-chords.ts";
export { createWheelZoomAccumulator } from "./lib/zoom-wheel.ts";
