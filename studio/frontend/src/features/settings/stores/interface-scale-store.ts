// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { setLayoutScale } from "@/lib/layout-scale";
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
// Floor is 50: browser Cmd/Ctrl+0 resets the browser zoom, and at 25% the undo row is ~3.5px.
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

export const INTERFACE_ZOOM_STEPS = [50, 67, 75, 80, 90, 100, 110, 125, 150, 175, 200] as const;

export function stepInterfaceScale(scale: number, direction: 1 | -1): number {
  const next =
    direction > 0
      ? INTERFACE_ZOOM_STEPS.find((step) => step > scale)
      : [...INTERFACE_ZOOM_STEPS].reverse().find((step) => step < scale);
  return next ?? scale;
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

export const INTERFACE_SCALE_VAR = "--ui-interface-scale";

/** Uses the UI tokens, not CSS `zoom`, which inflates viewport units. */
function applyWebInterfaceScale(scale: number): void {
  if (typeof document === "undefined") return;
  const zoom = interfaceScaleToZoom(scale);
  setLayoutScale(zoom);
  const style = document.documentElement.style;
  if (zoom === 1) style.removeProperty(INTERFACE_SCALE_VAR);
  else style.setProperty(INTERFACE_SCALE_VAR, String(zoom));
}

/** 1 on desktop, which uses webview zoom. */
export function webInterfaceScaleFactor(scale: number): number {
  return isTauri ? 1 : interfaceScaleToZoom(scale);
}

export function applyInterfaceScale(scale: number): Promise<void> {
  if (!isTauri) {
    applyWebInterfaceScale(scale);
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
    if (nextScale !== requestedInterfaceScale) {
      // a timed-out command can overwrite a newer native zoom when it finishes.
      appliedInterfaceScale = null;
      void applyInterfaceScale(requestedInterfaceScale).catch(() => undefined);
      return;
    }
    appliedInterfaceScale = nextScale;
    setAppliedInterfaceZoom(zoom);
  });
  interfaceScaleApplicationQueue = application.catch(() => undefined);
  return application;
}

/** Applying before first render avoids a 100% frame, but a hung Tauri IPC would blank the window,
 * so render at 100% past this deadline and let `provider.tsx` apply it later. */
export const INTERFACE_SCALE_FIRST_PAINT_TIMEOUT_MS = 1000;

export function applyInterfaceScaleBeforeFirstPaint(
  scale: number,
  timeoutMs: number = INTERFACE_SCALE_FIRST_PAINT_TIMEOUT_MS,
): Promise<void> {
  const applied = applyInterfaceScale(scale).catch(() => undefined);
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      // Cut the queue loose from the hung call, or every later scale change is dead until restart.
      interfaceScaleApplicationQueue = Promise.resolve();
      resolve();
    }, timeoutMs);
    void applied.finally(() => {
      clearTimeout(timer);
      resolve();
    });
  });
}
