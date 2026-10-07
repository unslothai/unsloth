// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * macOS draws the titlebar at a size webview zoom does not touch, so insets are divided by zoom.
 * Single source for provider.tsx's var() fallbacks and the tests.
 */
export const NATIVE_MAC_TITLEBAR_HEIGHT_PX = 34;
export const NATIVE_MAC_TRAFFIC_LIGHT_INSET_PX = 78;

export const NATIVE_MAC_TITLEBAR_HEIGHT_VAR = `var(--studio-native-titlebar-height, ${NATIVE_MAC_TITLEBAR_HEIGHT_PX}px)`;
export const NATIVE_MAC_TRAFFIC_LIGHT_INSET_VAR = `var(--studio-native-traffic-light-inset, ${NATIVE_MAC_TRAFFIC_LIGHT_INSET_PX}px)`;

let appliedInterfaceZoom = 1;
const appliedInterfaceZoomListeners = new Set<() => void>();

/** Page zoom is not observable from JS, so this is the last zoom handed to the webview. */
export function getAppliedInterfaceZoom(): number {
  return appliedInterfaceZoom;
}

export function subscribeAppliedInterfaceZoom(listener: () => void): () => void {
  appliedInterfaceZoomListeners.add(listener);
  return () => {
    appliedInterfaceZoomListeners.delete(listener);
  };
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
  for (const listener of appliedInterfaceZoomListeners) listener();
}
