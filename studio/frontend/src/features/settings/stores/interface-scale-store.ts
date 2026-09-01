// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { create } from "zustand";
import {
  type StateStorage,
  createJSONStorage,
  persist,
} from "zustand/middleware";
import {
  getAppliedInterfaceZoom,
  setAppliedInterfaceZoom,
} from "../lib/interface-scale-runtime.ts";

export { getAppliedInterfaceZoom };

export const INTERFACE_SCALE_STORAGE_KEY = "unsloth_interface_scale";
// The floor is 50, not the 25 Chrome and VS Code allow, because both of those ship
// Cmd/Ctrl+0 and this does not yet. At 25% the Settings row you would use to undo it
// renders around 3.5px, and the value is device-local in localStorage, so recovery means
// clearing app data. Drop it back to 25 once a reset accelerator exists.
export const INTERFACE_SCALE_RANGE = {
  min: 50,
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
let requestedInterfaceScale: number = INTERFACE_SCALE_RANGE.default;
let interfaceScaleApplicationQueue = Promise.resolve();

export function applyInterfaceScale(scale: number): Promise<void> {
  if (!isTauri) {
    return Promise.resolve();
  }
  requestedInterfaceScale = sanitizeInterfaceScale(scale);
  const application = interfaceScaleApplicationQueue.then(async () => {
    const nextScale = requestedInterfaceScale;
    if (nextScale === appliedInterfaceScale) {
      return;
    }
    const { getCurrentWebview } = await import("@tauri-apps/api/webview");
    const zoom = interfaceScaleToZoom(nextScale);
    await getCurrentWebview().setZoom(zoom);
    // A call abandoned at the first-paint deadline can still settle later, after a newer
    // scale has been asked for and applied. Committing then would report the stale zoom
    // as the live one.
    if (nextScale !== requestedInterfaceScale) {
      return;
    }
    appliedInterfaceScale = nextScale;
    setAppliedInterfaceZoom(zoom);
  });
  interfaceScaleApplicationQueue = application.catch(() => undefined);
  return application;
}

/**
 * Applying the scale before the first render is what stops a 100% frame painting and
 * then relaying out. Worth waiting for, not worth waiting forever: this is the only
 * thing gating first paint on the Tauri IPC bridge, and a rejection is caught while a
 * hang is not, so without a deadline a wedged bridge is a permanently blank window.
 *
 * Past the deadline the app renders at 100% and the effect in `provider.tsx` applies the
 * real scale whenever the bridge does answer.
 */
export const INTERFACE_SCALE_FIRST_PAINT_TIMEOUT_MS = 1000;

export function applyInterfaceScaleBeforeFirstPaint(
  scale: number,
  timeoutMs: number = INTERFACE_SCALE_FIRST_PAINT_TIMEOUT_MS,
): Promise<void> {
  const applied = applyInterfaceScale(scale).catch(() => undefined);
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      // Rendering is unblocked, but the queue is still chained to a call that may never
      // settle, and everything after it waits behind that. Cut it loose or the retry in
      // `provider.tsx` and every later scale change are dead until restart.
      interfaceScaleApplicationQueue = Promise.resolve();
      resolve();
    }, timeoutMs);
    void applied.finally(() => {
      clearTimeout(timer);
      resolve();
    });
  });
}
