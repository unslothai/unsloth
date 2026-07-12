// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ResolvedTheme } from "./theme-store";

export type ReduceMotionSetting = "system" | "on" | "off";

export type CustomModeColors = {
  accent: string | null;
  background: string | null;
  foreground: string | null;
};

export type ImportedFont = {
  /** Font-family name the FontFace is registered under. */
  name: string;
  /** data: URL of the uploaded font file. */
  dataUrl: string;
};

/**
 * Optional entries of the sidebar profile menu. Settings, Help, Log out, and
 * Shutdown are pinned and never appear here. The settings-tab shortcuts ship
 * hidden; General and About are covered by the pinned Settings and Help.
 */
export const SIDEBAR_MENU_ITEM_IDS = [
  "api",
  "darkMode",
  "guidedTour",
  "profile",
  "appearance",
  "resources",
  "chat",
  "connections",
] as const;

export type SidebarMenuItemId = (typeof SIDEBAR_MENU_ITEM_IDS)[number];

export type SidebarMenuItemPref = {
  id: SidebarMenuItemId;
  visible: boolean;
};

export const SIDEBAR_MENU_DEFAULT_VISIBLE: Record<SidebarMenuItemId, boolean> =
  {
    api: true,
    darkMode: true,
    guidedTour: true,
    profile: false,
    appearance: false,
    resources: false,
    chat: false,
    connections: false,
  };

export const MAX_IMPORTED_FONTS = 3;
/** ~1.5 MB file → ~2 MB base64; must stay in sync with the backend cap. */
export const MAX_IMPORTED_FONT_DATA_URL_LENGTH = 2_200_000;
/**
 * Aggregate cap across all imported fonts. localStorage quotas are commonly
 * ~5M UTF-16 units per origin; staying under that keeps the persisted store
 * writable even with other keys present.
 */
export const MAX_TOTAL_IMPORTED_FONT_DATA_URL_LENGTH = 4_400_000;

export type AppearanceCustomization = {
  colors: { light: CustomModeColors; dark: CustomModeColors };
  uiFont: string | null;
  headingFont: string | null;
  chatFont: string | null;
  codeFont: string | null;
  importedFonts: ImportedFont[];
  /** Root font size in px (rem base). null = browser default (16). */
  uiFontSize: number | null;
  /** Code/pre font size in px. null = inherit each element's own size. */
  codeFontSize: number | null;
  /** 0–100; 50 is neutral (no adjustment). */
  contrast: number;
  pointerCursors: boolean;
  reduceMotion: ReduceMotionSetting;
  /** true = the app default (antialiased). */
  fontSmoothing: boolean;
  /** true = content dissolves at panel edges; false = thin divider lines. */
  edgeFades: boolean;
  /** Order and visibility of the optional sidebar profile menu items. */
  sidebarMenu: SidebarMenuItemPref[];
};

const EMPTY_MODE_COLORS: CustomModeColors = {
  accent: null,
  background: null,
  foreground: null,
};

export const DEFAULT_CUSTOMIZATION: AppearanceCustomization = {
  colors: { light: { ...EMPTY_MODE_COLORS }, dark: { ...EMPTY_MODE_COLORS } },
  uiFont: null,
  headingFont: null,
  chatFont: null,
  codeFont: null,
  importedFonts: [],
  uiFontSize: null,
  codeFontSize: null,
  contrast: 50,
  pointerCursors: false,
  reduceMotion: "system",
  fontSmoothing: true,
  edgeFades: true,
  sidebarMenu: SIDEBAR_MENU_ITEM_IDS.map((id) => ({
    id,
    visible: SIDEBAR_MENU_DEFAULT_VISIBLE[id],
  })),
};

export const UI_FONT_SIZE_RANGE = { min: 12, max: 20, default: 16 } as const;
export const CODE_FONT_SIZE_RANGE = { min: 10, max: 20, default: 13 } as const;

const HEX_COLOR_PATTERN = /^#[0-9a-fA-F]{6}$/;

export function isHexColor(value: unknown): value is string {
  return typeof value === "string" && HEX_COLOR_PATTERN.test(value);
}

function sanitizeColor(value: unknown): string | null {
  return isHexColor(value) ? value.toLowerCase() : null;
}

function sanitizeFont(value: unknown): string | null {
  if (typeof value !== "string") return null;
  // Strip characters that could terminate the declaration or smuggle extra
  // CSS through the inline style (the value lands in style.setProperty).
  const cleaned = value
    .replace(/[;{}()<>"']/g, "")
    .trim()
    .slice(0, 200);
  return cleaned.length > 0 ? cleaned : null;
}

function sanitizeSize(
  value: unknown,
  range: { min: number; max: number },
): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  const rounded = Math.round(value);
  if (rounded < range.min || rounded > range.max) {
    return Math.min(range.max, Math.max(range.min, rounded));
  }
  return rounded;
}

function sanitizeModeColors(value: unknown): CustomModeColors {
  const source = (value ?? {}) as Partial<CustomModeColors>;
  return {
    accent: sanitizeColor(source.accent),
    background: sanitizeColor(source.background),
    foreground: sanitizeColor(source.foreground),
  };
}

const FONT_DATA_URL_PATTERN =
  /^data:(?:font\/(?:woff2?|ttf|otf|sfnt)|application\/(?:octet-stream|x-font-\w+|font-\w+));base64,[A-Za-z0-9+/=]+$/;

function sanitizeImportedFonts(value: unknown): ImportedFont[] {
  if (!Array.isArray(value)) return [];
  const fonts: ImportedFont[] = [];
  const seen = new Set<string>();
  let total = 0;
  for (const entry of value) {
    if (fonts.length >= MAX_IMPORTED_FONTS) break;
    const source = (entry ?? {}) as Partial<ImportedFont>;
    const name = sanitizeFont(source.name);
    if (!name || seen.has(name)) continue;
    const dataUrl = source.dataUrl;
    if (
      typeof dataUrl !== "string" ||
      dataUrl.length > MAX_IMPORTED_FONT_DATA_URL_LENGTH ||
      total + dataUrl.length > MAX_TOTAL_IMPORTED_FONT_DATA_URL_LENGTH ||
      !FONT_DATA_URL_PATTERN.test(dataUrl)
    ) {
      continue;
    }
    seen.add(name);
    total += dataUrl.length;
    fonts.push({ name, dataUrl });
  }
  return fonts;
}

function isSidebarMenuItemId(value: unknown): value is SidebarMenuItemId {
  return SIDEBAR_MENU_ITEM_IDS.includes(value as SidebarMenuItemId);
}

function sanitizeSidebarMenu(value: unknown): SidebarMenuItemPref[] {
  const items: SidebarMenuItemPref[] = [];
  const seen = new Set<SidebarMenuItemId>();
  for (const entry of Array.isArray(value) ? value : []) {
    const source = (entry ?? {}) as Partial<SidebarMenuItemPref>;
    if (!isSidebarMenuItemId(source.id) || seen.has(source.id)) continue;
    seen.add(source.id);
    items.push({ id: source.id, visible: source.visible !== false });
  }
  // Ids added after the payload was written land at the end with their
  // default visibility.
  for (const id of SIDEBAR_MENU_ITEM_IDS) {
    if (!seen.has(id))
      items.push({ id, visible: SIDEBAR_MENU_DEFAULT_VISIBLE[id] });
  }
  return items;
}

/**
 * Coerce arbitrary (persisted/remote) data into a valid customization object.
 * Anything malformed falls back to the default for that field, so a bad
 * payload can never wedge the UI.
 */
export function sanitizeCustomization(value: unknown): AppearanceCustomization {
  const source = (value ?? {}) as Partial<AppearanceCustomization> & {
    colors?: { light?: unknown; dark?: unknown };
  };
  const contrast =
    typeof source.contrast === "number" && Number.isFinite(source.contrast)
      ? Math.min(100, Math.max(0, Math.round(source.contrast)))
      : DEFAULT_CUSTOMIZATION.contrast;
  return {
    colors: {
      light: sanitizeModeColors(source.colors?.light),
      dark: sanitizeModeColors(source.colors?.dark),
    },
    uiFont: sanitizeFont(source.uiFont),
    headingFont: sanitizeFont(source.headingFont),
    chatFont: sanitizeFont(source.chatFont),
    codeFont: sanitizeFont(source.codeFont),
    importedFonts: sanitizeImportedFonts(source.importedFonts),
    uiFontSize: sanitizeSize(source.uiFontSize, UI_FONT_SIZE_RANGE),
    codeFontSize: sanitizeSize(source.codeFontSize, CODE_FONT_SIZE_RANGE),
    contrast,
    pointerCursors: source.pointerCursors === true,
    reduceMotion:
      source.reduceMotion === "on" || source.reduceMotion === "off"
        ? source.reduceMotion
        : "system",
    fontSmoothing: source.fontSmoothing !== false,
    edgeFades: source.edgeFades !== false,
    sidebarMenu: sanitizeSidebarMenu(source.sidebarMenu),
  };
}

export function isDefaultCustomization(c: AppearanceCustomization): boolean {
  return JSON.stringify(c) === JSON.stringify(DEFAULT_CUSTOMIZATION);
}

interface AppearanceCustomState {
  customization: AppearanceCustomization;
  setColor: (
    mode: ResolvedTheme,
    key: keyof CustomModeColors,
    value: string | null,
  ) => void;
  patch: (partial: Partial<AppearanceCustomization>) => void;
  addImportedFont: (font: ImportedFont) => void;
  removeImportedFont: (name: string) => void;
  replaceAll: (next: AppearanceCustomization) => void;
  resetAll: () => void;
}

export const useAppearanceCustomStore = create<AppearanceCustomState>()(
  persist(
    (set) => ({
      customization: DEFAULT_CUSTOMIZATION,
      setColor: (mode, key, value) =>
        set((state) => ({
          customization: {
            ...state.customization,
            colors: {
              ...state.customization.colors,
              [mode]: {
                ...state.customization.colors[mode],
                [key]: sanitizeColor(value),
              },
            },
          },
        })),
      patch: (partial) =>
        set((state) => ({
          customization: sanitizeCustomization({
            ...state.customization,
            ...partial,
          }),
        })),
      addImportedFont: (font) =>
        set((state) => ({
          customization: sanitizeCustomization({
            ...state.customization,
            importedFonts: [
              ...state.customization.importedFonts.filter(
                (f) => f.name !== font.name,
              ),
              font,
            ],
          }),
        })),
      removeImportedFont: (name) =>
        set((state) => {
          const c = state.customization;
          return {
            customization: sanitizeCustomization({
              ...c,
              importedFonts: c.importedFonts.filter((f) => f.name !== name),
              // Fall back to the default font wherever the removed one was in use.
              uiFont: c.uiFont === name ? null : c.uiFont,
              headingFont: c.headingFont === name ? null : c.headingFont,
              chatFont: c.chatFont === name ? null : c.chatFont,
              codeFont: c.codeFont === name ? null : c.codeFont,
            }),
          };
        }),
      replaceAll: (next) => set({ customization: sanitizeCustomization(next) }),
      resetAll: () => set({ customization: DEFAULT_CUSTOMIZATION }),
    }),
    {
      name: "unsloth_appearance_customization",
      version: 2,
      migrate: (persisted) => {
        const state = (persisted ?? {}) as Partial<AppearanceCustomState>;
        return {
          customization: sanitizeCustomization(state.customization),
        } as AppearanceCustomState;
      },
      // Sanitize on EVERY rehydrate, not just version bumps: a same-version
      // payload written by an older bundle (e.g. before importedFonts existed)
      // would otherwise reach the app with missing fields and crash .map calls.
      merge: (persisted, current) => ({
        ...current,
        customization: sanitizeCustomization(
          (persisted as Partial<AppearanceCustomState> | undefined)
            ?.customization,
        ),
      }),
    },
  ),
);

/* ------------------------------ DOM applier ------------------------------ */

const DEFAULT_SANS_STACK =
  '"Inter Variable", ui-sans-serif, sans-serif, system-ui';
const DEFAULT_HEADING_STACK =
  '"Hellix", "Space Grotesk Variable", var(--font-sans)';
const DEFAULT_MONO_STACK = "JetBrains Mono, monospace";

/** WCAG-ish relative luminance from a #rrggbb hex. */
function hexLuminance(hex: string): number {
  const channel = (i: number) => {
    const c = Number.parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
}

function readableForeground(hex: string): string {
  return hexLuminance(hex) > 0.45 ? "#111417" : "#ffffff";
}

/**
 * FontFaces registered for imported fonts, keyed by family name. Kept so a
 * removed import can be unregistered from document.fonts.
 */
const registeredFontFaces = new Map<string, FontFace>();

function syncImportedFonts(fonts: ImportedFont[]): void {
  if (typeof document === "undefined" || !("fonts" in document)) return;
  // Never trust the shape at this boundary: a stale
  // persisted payload without importedFonts must not crash the applier.
  const wanted = new Map(
    (Array.isArray(fonts) ? fonts : []).map((f) => [f.name, f.dataUrl]),
  );
  for (const [name, face] of registeredFontFaces) {
    if (!wanted.has(name)) {
      document.fonts.delete(face);
      registeredFontFaces.delete(name);
    }
  }
  for (const [name, dataUrl] of wanted) {
    if (registeredFontFaces.has(name)) continue;
    try {
      const face = new FontFace(name, `url(${dataUrl})`);
      registeredFontFaces.set(name, face);
      document.fonts.add(face);
      face.load().catch(() => {
        document.fonts.delete(face);
        registeredFontFaces.delete(name);
      });
    } catch {
      registeredFontFaces.delete(name);
    }
  }
}

/**
 * The custom "Accent" recolors the accent family (toggles, badges, chart-1).
 * Focus/selection rings and button colors (--primary) are deliberately left
 * alone: highlight borders stay neutral and Classic's buttons stay neutral.
 */
const ACCENT_VARS = ["--control-accent", "--chart-1"] as const;
const ACCENT_FG_VARS = ["--control-accent-foreground"] as const;

/**
 * Push the customization onto <html> as inline CSS variables, attributes, and
 * classes. Everything is keyed off explicit hooks (inline vars beat every
 * palette block; the classes/attributes gate rules in index.css), so the
 * default customization leaves the document byte-identical to stock.
 */
export function applyCustomizationToDocument(
  c: AppearanceCustomization,
  resolved: ResolvedTheme,
): void {
  if (typeof document === "undefined") return;
  const el = document.documentElement;
  const style = el.style;

  const setVar = (name: string, value: string | null) => {
    if (value === null) style.removeProperty(name);
    else style.setProperty(name, value);
  };

  const colors = c.colors[resolved];

  for (const name of ACCENT_VARS) setVar(name, colors.accent);
  for (const name of ACCENT_FG_VARS) {
    setVar(name, colors.accent ? readableForeground(colors.accent) : null);
  }
  setVar("--background", colors.background);
  setVar("--foreground", colors.foreground);

  syncImportedFonts(c.importedFonts);

  // Family names are single identifiers picked from the dropdown (sanitizeFont
  // strips quote characters), so quoting here is always safe.
  setVar(
    "--font-sans",
    c.uiFont ? `"${c.uiFont}", ${DEFAULT_SANS_STACK}` : null,
  );
  setVar(
    "--font-heading",
    c.headingFont ? `"${c.headingFont}", ${DEFAULT_HEADING_STACK}` : null,
  );
  setVar(
    "--font-mono",
    c.codeFont ? `"${c.codeFont}", ${DEFAULT_MONO_STACK}` : null,
  );

  if (c.chatFont) {
    el.setAttribute("data-chat-font", "");
    setVar("--custom-chat-font", `"${c.chatFont}", ${DEFAULT_SANS_STACK}`);
  } else {
    el.removeAttribute("data-chat-font");
    setVar("--custom-chat-font", null);
  }

  if (c.uiFontSize !== null && c.uiFontSize !== UI_FONT_SIZE_RANGE.default) {
    style.fontSize = `${c.uiFontSize}px`;
  } else {
    style.removeProperty("font-size");
  }

  if (c.codeFontSize !== null) {
    el.setAttribute("data-code-font-size", "");
    setVar("--custom-code-font-size", `${c.codeFontSize}px`);
  } else {
    el.removeAttribute("data-code-font-size");
    setVar("--custom-code-font-size", null);
  }

  if (c.contrast !== 50) {
    // Map |contrast - 50| ∈ (0, 50] onto a 0–40% color-mix toward the
    // foreground (higher contrast) or background (lower contrast).
    const mix = Math.round(Math.abs(c.contrast - 50) * 0.8);
    el.setAttribute("data-contrast-adjust", "");
    setVar("--contrast-mix", `${mix}%`);
    setVar(
      "--contrast-target",
      c.contrast > 50 ? "var(--foreground)" : "var(--background)",
    );
  } else {
    el.removeAttribute("data-contrast-adjust");
    setVar("--contrast-mix", null);
    setVar("--contrast-target", null);
  }

  el.classList.toggle("pointer-cursors", c.pointerCursors);
  el.classList.toggle("force-reduced-motion", c.reduceMotion === "on");
  // "off" opts out of the OS reduced-motion preference for CSS animations;
  // the media rules in index.css skip html.force-motion.
  el.classList.toggle("force-motion", c.reduceMotion === "off");
  el.classList.toggle("no-font-smoothing", !c.fontSmoothing);
  // Off swaps the scroll-edge dissolves for thin divider lines (index.css
  // and hub.css key their fade rules off this class).
  el.classList.toggle("no-edge-fades", !c.edgeFades);
}
