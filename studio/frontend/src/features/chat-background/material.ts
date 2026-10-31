// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports
import { isMacPlatform } from "@/features/settings/lib/keyboard-shortcuts";
import { isTauri } from "@/lib/api-base";
import { useEffect } from "react";
import { create } from "zustand";
import {
  type StateStorage,
  createJSONStorage,
  persist,
} from "zustand/middleware";
import {
  type WindowMaterial,
  sanitizeMaterial,
  sanitizeStrength,
} from "./model";

export const MATERIAL_STORAGE_KEY = "unsloth_window_material";

/** Per device, not synced: the macOS material means nothing in another browser. */
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
      // stays in memory for this session
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

export function nativeMaterialAvailable(): boolean {
  return isTauri && isMacPlatform();
}

export const useMaterialStore = create<{
  material: WindowMaterial;
  /** The desktop shell refused the native material (an older Unsloth Desktop). */
  nativeFailed: boolean;
  /** Dropdowns, menus and pickers take the glass too. */
  menus: boolean;
  strength: number;
  setMaterial: (material: WindowMaterial) => void;
  setMenus: (menus: boolean) => void;
  setStrength: (strength: number) => void;
}>()(
  persist(
    (set) => ({
      material: "solid",
      nativeFailed: false,
      menus: false,
      strength: 50,
      setMaterial: (material) =>
        set({ material: sanitizeMaterial(material), nativeFailed: false }),
      setMenus: (menus) => set({ menus }),
      setStrength: (strength) => set({ strength: sanitizeStrength(strength) }),
    }),
    {
      name: MATERIAL_STORAGE_KEY,
      storage: createJSONStorage(() => guardedLocalStorage),
      partialize: (s) => ({
        material: s.material,
        menus: s.menus,
        strength: s.strength,
      }),
      merge: (persisted, current) => {
        const stored = persisted as
          | {
              material?: unknown;
              menus?: unknown;
              strength?: unknown;
            }
          | undefined;
        return {
          ...current,
          material: sanitizeMaterial(stored?.material),
          menus: stored?.menus === true,
          strength: sanitizeStrength(stored?.strength),
        };
      },
    },
  ),
);

async function setNativeVibrancy(enabled: boolean): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("set_native_vibrancy", { enabled });
}

let nativeApplied = false;

/** Mirrors the material onto <html data-glass>; the macOS one waits for the shell to apply it. */
export function WindowMaterialMount() {
  const material = useMaterialStore((s) => s.material);
  const menus = useMaterialStore((s) => s.menus);
  const strength = useMaterialStore((s) => s.strength);
  useEffect(() => {
    // 0-100 onto a 0.2-2x multiplier of each material's tint and blur, 1x at 50.
    const root = document.documentElement;
    const factor =
      strength <= 50 ? 0.2 + (0.8 * strength) / 50 : 1 + (strength - 50) / 50;
    if (strength === 50) root.style.removeProperty("--glass-strength");
    else root.style.setProperty("--glass-strength", factor.toFixed(3));
  }, [strength]);
  useEffect(() => {
    // The menu rules also require data-glass, so a refused macOS material leaves menus solid.
    document.documentElement.toggleAttribute(
      "data-glass-menus",
      menus && material !== "solid",
    );
  }, [menus, material]);
  useEffect(() => {
    const root = document.documentElement;
    const wantsNative = material === "macos" && nativeMaterialAvailable();
    let cancelled = false;
    const show = (value: WindowMaterial) => {
      if (value === "solid") root.removeAttribute("data-glass");
      else root.setAttribute("data-glass", value);
    };
    if (!wantsNative) {
      show(material === "macos" ? "solid" : material);
      if (nativeApplied) {
        nativeApplied = false;
        void setNativeVibrancy(false).catch(() => undefined);
      }
      return;
    }
    // A transparent page over an opaque window would show the webview's blank fill.
    void setNativeVibrancy(true)
      .then((applied) => {
        if (cancelled) return;
        if (!applied) throw new Error("unsupported");
        nativeApplied = true;
        show("macos");
      })
      .catch(() => {
        if (cancelled) return;
        show("solid");
        useMaterialStore.setState({ nativeFailed: true });
      });
    return () => {
      cancelled = true;
    };
  }, [material]);
  return null;
}
