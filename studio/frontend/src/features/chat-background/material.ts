// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports
import { isMacPlatform } from "@/features/settings/lib/keyboard-shortcuts";
// eslint-disable-next-line no-restricted-imports
import { useTheme } from "@/features/settings/stores/theme-store";
import { isTauri } from "@/lib/api-base";
import { useEffect, useState } from "react";
import { create } from "zustand";
import {
  type StateStorage,
  createJSONStorage,
  persist,
} from "zustand/middleware";
import {
  type BackgroundRoute,
  type BackgroundScope,
  DEFAULT_GLASS_STRENGTH,
  type GlassStrengths,
  type LensMode,
  type WindowMaterial,
  inScope,
  sanitizeLens,
  sanitizeMaterial,
  sanitizeScope,
  sanitizeStrength,
  sanitizeStrengths,
} from "./model";
import { useChatWelcomeShown } from "./welcome";

export const MATERIAL_STORAGE_KEY = "unsloth_window_material";

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
  lens: LensMode;
  scope: BackgroundScope;
  /** Not persisted: whether the current route is inside scope, set by WindowMaterialMount. */
  here: boolean;
  setMaterial: (material: WindowMaterial) => void;
  setEnabled: (enabled: boolean) => void;
  setMenus: (menus: boolean) => void;
  setStrength: (strength: number, mode: keyof GlassStrengths) => void;
  setLens: (lens: LensMode) => void;
  setScope: (scope: BackgroundScope) => void;
}>()(
  persist(
    (set) => ({
      material: "solid",
      style: "liquid-3d",
      nativeFailed: false,
      menus: true,
      strength: DEFAULT_GLASS_STRENGTH,
      lens: "off",
      scope: "all",
      here: true,
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
      setLens: (lens) => set({ lens: sanitizeLens(lens) }),
      setScope: (scope) => set({ scope: sanitizeScope(scope, "all") }),
    }),
    {
      name: MATERIAL_STORAGE_KEY,
      storage: createJSONStorage(() => guardedLocalStorage),
      partialize: (s) => ({
        material: s.material,
        style: s.style,
        menus: s.menus,
        strength: s.strength,
        lens: s.lens,
        scope: s.scope,
      }),
      merge: (persisted, current) => {
        const stored = persisted as
          | {
              material?: unknown;
              style?: unknown;
              menus?: unknown;
              strength?: unknown;
              lens?: unknown;
              scope?: unknown;
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
          lens: sanitizeLens(stored?.lens),
          // Glass applied everywhere before the setting existed.
          scope: sanitizeScope(stored?.scope, "all"),
        };
      },
    },
  ),
);

async function setNativeVibrancy(enabled: boolean): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("set_native_vibrancy", { enabled });
}

/** The material drawn on this route: solid outside the glass scope. */
export function useAppliedMaterial(): WindowMaterial {
  return useMaterialStore((s) => (s.here ? s.material : "solid"));
}

export function useGlassStrength(): number {
  const { resolved } = useTheme();
  return useMaterialStore((s) => s.strength[resolved]);
}

let nativeApplied = false;

// WebKit repaints a glass surface's blurred shadows and backdrop whenever anything inside it
// changes, and on CPU painting that dominates scrolling and typing. Its children get their own
// layers there (index.css, data-glass-layers); Chromium composites these already.
const WEBKIT_LAYERS =
  typeof navigator !== "undefined" &&
  /AppleWebKit/.test(navigator.userAgent) &&
  !/Chrom|Edg|Android/i.test(navigator.userAgent);

export function WindowMaterialMount({ route }: { route?: BackgroundRoute }) {
  const material = useMaterialStore((s) => s.material);
  const menus = useMaterialStore((s) => s.menus);
  const scope = useMaterialStore((s) => s.scope);
  const welcome = useChatWelcomeShown();
  const strength = useGlassStrength();
  // What the native shell settled on; the attribute also needs this route to be in scope.
  const [shown, setShown] = useState<WindowMaterial>("solid");
  const here =
    !route || inScope(scope, route === "chat" && welcome ? "newChat" : route);
  useEffect(() => {
    useMaterialStore.setState({ here });
  }, [here]);
  useEffect(() => {
    const root = document.documentElement;
    const applied = shown === "solid" || !here ? null : shown;
    // Transparent is the plain theme with see-through fills, so it takes none of the frosted glass rules.
    if (applied && applied !== "transparent") root.setAttribute("data-glass", applied);
    else root.removeAttribute("data-glass");
    root.toggleAttribute("data-glass-clear", applied === "transparent");
  }, [shown, here]);
  useEffect(() => {
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
    if (material === "transparent") {
      root.style.setProperty("--glass-fill-alpha", `${strength}%`);
    } else {
      root.style.removeProperty("--glass-fill-alpha");
    }
  }, [strength, material]);
  useEffect(() => {
    document.documentElement.toggleAttribute(
      "data-glass-menus",
      menus && material !== "solid" && material !== "transparent" && here,
    );
  }, [menus, material, here]);
  useEffect(() => {
    document.documentElement.toggleAttribute(
      "data-glass-layers",
      WEBKIT_LAYERS && material !== "solid" && material !== "transparent" && here,
    );
  }, [material, here]);
  useEffect(() => {
    const root = document.documentElement;
    const wantsNative = material === "macos" && nativeMaterialAvailable();
    let cancelled = false;
    if (!wantsNative) {
      setShown(material === "macos" ? "solid" : material);
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
        setShown("macos");
      })
      .catch(() => {
        if (cancelled) return;
        setShown("solid");
        useMaterialStore.setState({ nativeFailed: true });
      });
    return () => {
      cancelled = true;
    };
  }, [material]);
  return null;
}
