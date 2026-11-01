// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports
import { isMacPlatform } from "@/features/settings/lib/keyboard-shortcuts";
// eslint-disable-next-line no-restricted-imports
import { useTheme } from "@/features/settings/stores/theme-store";
import { isTauri } from "@/lib/api-base";
import { useEffect } from "react";
import { create } from "zustand";
import {
  type StateStorage,
  createJSONStorage,
  persist,
} from "zustand/middleware";
import {
  DEFAULT_GLASS_STRENGTH,
  type GlassStrengths,
  type WindowMaterial,
  sanitizeMaterial,
  sanitizeStrength,
  sanitizeStrengths,
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

export type GlassStyle = Exclude<WindowMaterial, "solid">;
const sanitizeStyle = (value: unknown): GlassStyle | null => {
  const material = sanitizeMaterial(value);
  return material === "solid" ? null : material;
};

export const useMaterialStore = create<{
  material: WindowMaterial;
  style: GlassStyle;
  nativeFailed: boolean;
  menus: boolean;
  strength: GlassStrengths;
  setMaterial: (material: WindowMaterial) => void;
  setEnabled: (enabled: boolean) => void;
  setMenus: (menus: boolean) => void;
  setStrength: (strength: number, mode: keyof GlassStrengths) => void;
}>()(
  persist(
    (set) => ({
      material: "solid",
      style: "liquid-3d",
      nativeFailed: false,
      menus: true,
      strength: DEFAULT_GLASS_STRENGTH,
      setMaterial: (material) => {
        const next = sanitizeMaterial(material);
        set(
          next === "solid"
            ? { material: next, nativeFailed: false }
            : { material: next, style: next, nativeFailed: false },
        );
      },
      setEnabled: (enabled) =>
        set((s) => ({
          material: enabled ? s.style : "solid",
          nativeFailed: false,
        })),
      setMenus: (menus) => set({ menus }),
      setStrength: (strength, mode) =>
        set((s) => ({
          strength: { ...s.strength, [mode]: sanitizeStrength(strength) },
        })),
    }),
    {
      name: MATERIAL_STORAGE_KEY,
      storage: createJSONStorage(() => guardedLocalStorage),
      partialize: (s) => ({
        material: s.material,
        style: s.style,
        menus: s.menus,
        strength: s.strength,
      }),
      merge: (persisted, current) => {
        const stored = persisted as
          | {
              material?: unknown;
              style?: unknown;
              menus?: unknown;
              strength?: unknown;
            }
          | undefined;
        return {
          ...current,
          material: sanitizeMaterial(stored?.material),
          style:
            sanitizeStyle(stored?.style) ??
            sanitizeStyle(stored?.material) ??
            "liquid-3d",
          menus: typeof stored?.menus === "boolean" ? stored.menus : true,
          strength: sanitizeStrengths(stored?.strength),
        };
      },
    },
  ),
);

async function setNativeVibrancy(enabled: boolean): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("set_native_vibrancy", { enabled });
}

export function useGlassStrength(): number {
  const { resolved } = useTheme();
  return useMaterialStore((s) => s.strength[resolved]);
}

let nativeApplied = false;

export function WindowMaterialMount() {
  const material = useMaterialStore((s) => s.material);
  const menus = useMaterialStore((s) => s.menus);
  const strength = useGlassStrength();
  useEffect(() => {
    // 0-100 onto multipliers, 1x at 50, applied after the readability floors so the setting always shows.
    const root = document.documentElement;
    const blur =
      strength <= 50 ? 0.2 + (0.8 * strength) / 50 : 1 + (strength - 50) / 50;
    const opacity =
      strength <= 50 ? 0.35 + (0.65 * strength) / 50 : 1 + (strength - 50) / 50;
    if (material === "solid" || strength === 50) {
      root.style.removeProperty("--glass-strength");
      root.style.removeProperty("--glass-opacity");
    } else {
      root.style.setProperty("--glass-strength", blur.toFixed(3));
      root.style.setProperty("--glass-opacity", opacity.toFixed(3));
    }
  }, [strength, material]);
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
        if (cancelled) {
          // Switched away while the shell was applying it: undo a late success unless macOS is back.
          if (!applied) return;
          if (useMaterialStore.getState().material === "macos") nativeApplied = true;
          else void setNativeVibrancy(false).catch(() => undefined);
          return;
        }
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
