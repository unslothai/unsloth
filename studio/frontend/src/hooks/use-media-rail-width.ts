// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CSSProperties } from "react";
import { useUiSpaceScale } from "./use-ui-space-scale";
import { createPanelWidthStore } from "./use-panel-width.ts";

/** Below this the sliders and their value pills start colliding. */
export const MEDIA_RAIL_WIDTH_MIN = 320;
export const MEDIA_RAIL_WIDTH_MAX = 760;

/** Settings rail width per media page, in the same units as the former fixed width. */
const stores = {
  images: createPanelWidthStore({
    key: "images_rail_width",
    min: MEDIA_RAIL_WIDTH_MIN,
    max: MEDIA_RAIL_WIDTH_MAX,
    fallback: 408,
    maxViewportFraction: 0.6,
  }),
  video: createPanelWidthStore({
    key: "video_rail_width",
    min: MEDIA_RAIL_WIDTH_MIN,
    max: MEDIA_RAIL_WIDTH_MAX,
    fallback: 400,
    maxViewportFraction: 0.6,
  }),
  audio: createPanelWidthStore({
    key: "audio_rail_width",
    min: MEDIA_RAIL_WIDTH_MIN,
    max: MEDIA_RAIL_WIDTH_MAX,
    fallback: 408,
    maxViewportFraction: 0.6,
  }),
};

export type MediaRailKind = keyof typeof stores;

/** Marks the element that owns `--media-rail-width` for its header and rail. */
export const MEDIA_RAIL_ROOT_ATTR = "data-media-rail-root";

export function useMediaRailWidth(kind: MediaRailKind) {
  const store = stores[kind];
  const rail = store.useWidth();
  // Includes the interface scale, and matches the rail's former calc(408px * --ui-space-scale).
  const scale = useUiSpaceScale();
  return {
    ...rail,
    scale,
    clamp: store.clamp,
    rootStyle: { "--media-rail-width": `${rail.width * scale}px` } as CSSProperties,
  };
}
