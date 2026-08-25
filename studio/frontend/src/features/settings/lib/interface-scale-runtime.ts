// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const NATIVE_MAC_TITLEBAR_HEIGHT = 34;
const NATIVE_MAC_TRAFFIC_LIGHT_INSET = 78;

let appliedInterfaceZoom = 1;

export function getAppliedInterfaceZoom(): number {
  return appliedInterfaceZoom;
}

export function setAppliedInterfaceZoom(zoom: number): void {
  appliedInterfaceZoom = zoom;
  document.documentElement.style.setProperty(
    "--studio-native-titlebar-height",
    `${NATIVE_MAC_TITLEBAR_HEIGHT / zoom}px`,
  );
  document.documentElement.style.setProperty(
    "--studio-native-traffic-light-inset",
    `${NATIVE_MAC_TRAFFIC_LIGHT_INSET / zoom}px`,
  );
}
