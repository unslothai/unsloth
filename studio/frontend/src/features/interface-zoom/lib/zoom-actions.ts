// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  stepInterfaceScale,
  useInterfaceScaleStore,
} from "@/features/settings";
import { create } from "zustand";
import type { ZoomDirection } from "./zoom-chords.ts";

interface ZoomPopupState {
  token: number;
  open: boolean;
  show: () => void;
  hide: () => void;
}

export const useZoomPopupStore = create<ZoomPopupState>()((set) => ({
  token: 0,
  open: false,
  show: () => set((s) => ({ open: true, token: s.token + 1 })),
  hide: () => set({ open: false }),
}));

export function zoomInterface(direction: ZoomDirection): void {
  const scale = useInterfaceScaleStore.getState();
  if (direction === 0) scale.reset();
  else scale.setScale(stepInterfaceScale(scale.scale, direction));
  useZoomPopupStore.getState().show();
}

export type ZoomScope = {
  contains: (element: Element) => boolean;
  zoom: (direction: ZoomDirection) => void;
};

const scopes = new Set<ZoomScope>();

export function registerZoomScope(scope: ZoomScope): () => void {
  scopes.add(scope);
  return () => {
    scopes.delete(scope);
  };
}

export function zoomScopeFor(target: EventTarget | null): ZoomScope | null {
  if (!(target instanceof Element)) return null;
  for (const scope of scopes) if (scope.contains(target)) return scope;
  return null;
}

// Dedupes a macOS View menu echo of a chord the page already handled.
const MENU_ECHO_MS = 150;
let lastZoom: { source: "chord" | "menu"; at: number } | null = null;

function echoed(source: "chord" | "menu"): boolean {
  const now = performance.now();
  if (lastZoom && lastZoom.source !== source && now - lastZoom.at < MENU_ECHO_MS) return true;
  lastZoom = { source, at: now };
  return false;
}

export function zoomInterfaceFromChord(direction: ZoomDirection): void {
  if (!echoed("chord")) zoomInterface(direction);
}

export function zoomScopeFromChord(scope: ZoomScope, direction: ZoomDirection): void {
  if (!echoed("chord")) scope.zoom(direction);
}

export function zoomInterfaceFromMenu(direction: ZoomDirection): void {
  if (echoed("menu")) return;
  const scope = zoomScopeFor(document.activeElement);
  if (scope) scope.zoom(direction);
  else zoomInterface(direction);
}
