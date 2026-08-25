// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { create } from "zustand";
import {
  type StateStorage,
  createJSONStorage,
  persist,
} from "zustand/middleware";

export const INTERFACE_SCALE_STORAGE_KEY = "unsloth_interface_scale";
export const INTERFACE_SCALE_RANGE = {
  min: 25,
  max: 200,
  default: 100,
} as const;

const guardedLocalStorage: StateStorage = {
  getItem: (name) => {
    try {
      return window.localStorage.getItem(name);
    } catch {
      return null;
    }
  },
  setItem: (name, value) => {
    try {
      window.localStorage.setItem(name, value);
    } catch {
      // ignore: the scale stays in memory for this session
    }
  },
  removeItem: (name) => {
    try {
      window.localStorage.removeItem(name);
    } catch {
      // ignore
    }
  },
};

export function sanitizeInterfaceScale(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return INTERFACE_SCALE_RANGE.default;
  }
  return Math.min(
    INTERFACE_SCALE_RANGE.max,
    Math.max(INTERFACE_SCALE_RANGE.min, Math.round(value)),
  );
}

export function interfaceScaleToZoom(scale: number): number {
  return sanitizeInterfaceScale(scale) / 100;
}

interface InterfaceScaleState {
  scale: number;
  setScale: (scale: number) => void;
  reset: () => void;
}

export const useInterfaceScaleStore = create<InterfaceScaleState>()(
  persist(
    (set) => ({
      scale: INTERFACE_SCALE_RANGE.default,
      setScale: (scale) => set({ scale: sanitizeInterfaceScale(scale) }),
      reset: () => set({ scale: INTERFACE_SCALE_RANGE.default }),
    }),
    {
      name: INTERFACE_SCALE_STORAGE_KEY,
      storage: createJSONStorage(() => guardedLocalStorage),
      merge: (persisted, current) => ({
        ...current,
        scale: sanitizeInterfaceScale(
          (persisted as Partial<InterfaceScaleState> | undefined)?.scale,
        ),
      }),
    },
  ),
);

let appliedInterfaceScale: number | null = null;

export async function applyInterfaceScale(scale: number): Promise<void> {
  if (!isTauri) {
    return;
  }
  const nextScale = sanitizeInterfaceScale(scale);
  if (nextScale === appliedInterfaceScale) {
    return;
  }
  const { getCurrentWebview } = await import("@tauri-apps/api/webview");
  await getCurrentWebview().setZoom(interfaceScaleToZoom(nextScale));
  appliedInterfaceScale = nextScale;
}
