// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * macOS draws the titlebar and traffic lights itself, at a size webview zoom does not
 * touch. Every CSS inset that has to clear them is therefore divided by the zoom, so the
 * gap stays the same number of screen points at any interface scale.
 *
 * These two numbers are the single source for that. `provider.tsx` builds its `var(...)`
 * fallbacks from them and the tests assert against them, so changing the titlebar height
 * here cannot leave a stale divisor behind at any scale but 100%.
 */
export const NATIVE_MAC_TITLEBAR_HEIGHT_PX = 34;
export const NATIVE_MAC_TRAFFIC_LIGHT_INSET_PX = 78;

export const NATIVE_MAC_TITLEBAR_HEIGHT_VAR = `var(--studio-native-titlebar-height, ${NATIVE_MAC_TITLEBAR_HEIGHT_PX}px)`;
export const NATIVE_MAC_TRAFFIC_LIGHT_INSET_VAR = `var(--studio-native-traffic-light-inset, ${NATIVE_MAC_TRAFFIC_LIGHT_INSET_PX}px)`;

let appliedInterfaceZoom = 1;

/**
 * The zoom last handed to the webview, not a reading off it. Page zoom is not observable
 * from JS, so this is the only value the non-Windows drop path has.
 */
export function getAppliedInterfaceZoom(): number {
  return appliedInterfaceZoom;
}

export function setAppliedInterfaceZoom(zoom: number): void {
  appliedInterfaceZoom = zoom;
  document.documentElement.style.setProperty(
    "--studio-native-titlebar-height",
    `${NATIVE_MAC_TITLEBAR_HEIGHT_PX / zoom}px`,
  );
  document.documentElement.style.setProperty(
    "--studio-native-traffic-light-inset",
    `${NATIVE_MAC_TRAFFIC_LIGHT_INSET_PX / zoom}px`,
  );
}
