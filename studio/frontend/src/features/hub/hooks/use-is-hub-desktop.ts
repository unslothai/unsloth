// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

const HUB_DESKTOP_QUERY = "(min-width: 1024px)";

function getSnapshot(): boolean {
  if (typeof window === "undefined") return true;
  return window.matchMedia(HUB_DESKTOP_QUERY).matches;
}

function subscribe(callback: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const mql = window.matchMedia(HUB_DESKTOP_QUERY);
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

export function useIsHubDesktop(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => true);
}
