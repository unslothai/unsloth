// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  type StateStorage,
  createJSONStorage,
  persist,
} from "zustand/middleware";
import { COLOR_THEMES, type ColorThemeId } from "../lib/color-themes.ts";
import type { ResolvedTheme } from "./theme-store";

// localStorage can throw (private browsing) and zustand's persist write is unguarded, so swallow.
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
      // ignore: the customization stays in memory for this session
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

export type ReduceMotionSetting = "system" | "on" | "off";
export type ChatWidthSetting = "standard" | "wide" | "full";
export type SentAttachmentsSetting = "auto" | "list" | "chips";

export type CustomModeColors = {
  accent: string | null;
  background: string | null;
  foreground: string | null;
};

export type ImportedFont = {
  name: string;
  dataUrl: string;
};

/** Settings, Help, Log out and Shutdown are pinned and never appear here. */
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

/** Array order is render order; unpinned rows go to "More" except a lone one, which is hidden. */
export const SIDEBAR_NAV_ITEM_IDS = [
  "hub",
  "projects",
  "library",
  "images",
  "video",
  "audio",
  "train",
  "recipes",
  "export",
  "api",
] as const;

export type SidebarNavItemId = (typeof SIDEBAR_NAV_ITEM_IDS)[number];

export type SidebarNavItemPref = {
  id: SidebarNavItemId;
  pinned: boolean;
};

export const SIDEBAR_NAV_AUTO_ITEM_IDS = ["projects"] as const satisfies
  readonly SidebarNavItemId[];

/** Projects steps into "More" while the Projects section lists folders, until a toggle in
 * Customize sidebar drops it from `auto`. */
export function sidebarNavRowPinned(
  item: SidebarNavItemPref,
  auto: readonly SidebarNavItemId[],
  context: { projectsSectionShowing: boolean },
): boolean {
  if (item.id === "projects" && auto.includes("projects")) {
    return !context.projectsSectionShowing;
  }
  return item.pinned;
}

export function sidebarNavAutoAfterChoice(
  auto: readonly SidebarNavItemId[],
  id: SidebarNavItemId,
): SidebarNavItemId[] {
  return auto.filter((entry) => entry !== id);
}

// Matches the shipped layout, so an untouched install looks unchanged.
export const SIDEBAR_NAV_DEFAULT_PINNED: Record<SidebarNavItemId, boolean> = {
  hub: true,
  projects: true,
  library: true,
  images: true,
  video: false,
  audio: false,
  train: true,
  recipes: false,
  export: false,
  api: false,
};

/** Every previously shipped layout, so a migration can tell an untouched install from an
 * arranged one. */
const SHIPPED_SIDEBAR_NAV_DEFAULTS: SidebarNavItemPref[][] = [
  [
    { id: "projects", pinned: true },
    { id: "hub", pinned: true },
    { id: "images", pinned: true },
    { id: "train", pinned: true },
    { id: "video", pinned: false },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
  ],
  [
    { id: "projects", pinned: true },
    { id: "hub", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: true },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: true },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "audio", pinned: false },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: true },
    { id: "audio", pinned: false },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ],
];

export const MAX_IMPORTED_FONTS = 3;
/** Must match the backend name max_length (100). */
export const MAX_IMPORTED_FONT_NAME_LENGTH = 100;
/** ~1.5 MB file → ~2 MB base64; must stay in sync with the backend cap. */
export const MAX_IMPORTED_FONT_DATA_URL_LENGTH = 2_200_000;
/** Keeps the persisted store under typical ~5M-unit localStorage quotas. */
export const MAX_TOTAL_IMPORTED_FONT_DATA_URL_LENGTH = 4_400_000;

export type AppearanceCustomization = {
  colors: { light: CustomModeColors; dark: CustomModeColors };
  uiFont: string | null;
  headingFont: string | null;
  chatFont: string | null;
  chatWidth: ChatWidthSetting;
  sentAttachments: SentAttachmentsSetting;
  codeFont: string | null;
  importedFonts: ImportedFont[];
  /** null = app default (15). */
  uiFontSize: number | null;
  /** null = inherit each element's own size. */
  codeFontSize: number | null;
  /** 0–100; 50 is neutral. */
  contrast: number;
  pointerCursors: boolean;
  reduceMotion: ReduceMotionSetting;
  /** true = the app default (antialiased). */
  fontSmoothing: boolean;
  sidebarMenu: SidebarMenuItemPref[];
  sidebarNav: SidebarNavItemPref[];
  /** Rows still following their automatic rule; only Projects has one. */
  sidebarNavAuto: SidebarNavItemId[];
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
  chatWidth: "standard",
  sentAttachments: "auto",
  codeFont: null,
  importedFonts: [],
  uiFontSize: null,
  codeFontSize: null,
  contrast: 50,
  pointerCursors: false,
  reduceMotion: "system",
  fontSmoothing: true,
  sidebarMenu: SIDEBAR_MENU_ITEM_IDS.map((id) => ({
    id,
    visible: SIDEBAR_MENU_DEFAULT_VISIBLE[id],
  })),
  sidebarNav: SIDEBAR_NAV_ITEM_IDS.map((id) => ({
    id,
    pinned: SIDEBAR_NAV_DEFAULT_PINNED[id],
  })),
  sidebarNavAuto: [...SIDEBAR_NAV_AUTO_ITEM_IDS],
};

export const UI_FONT_SIZE_RANGE = { min: 12, max: 20, default: 15 } as const;
export const CODE_FONT_SIZE_RANGE = { min: 10, max: 20, default: 12 } as const;
const UI_FONT_SIZE_CSS_BASE = 16;

/** Read by `html[data-contrast-adjust]` in index.css; exported so tests name the same vars. */
export const CONTRAST_SURFACE_MIX_VAR = "--contrast-surface-mix";
export const CONTRAST_LINE_MIX_VAR = "--contrast-line-mix";
export const CONTRAST_CONTROL_MIX_VAR = "--contrast-control-mix";
export const CONTRAST_FILL_MIX_VAR = "--contrast-fill-mix";
export const CONTRAST_STATE_MIX_VAR = "--contrast-state-mix";
export const CONTRAST_TEXT_MIX_VAR = "--contrast-text-mix";
export const CONTRAST_INK_MIX_VAR = "--contrast-ink-mix";
export const CONTRAST_INK_TARGET_VAR = "--contrast-ink-target";
export const CONTRAST_PANEL_INK_TARGET_VAR = "--contrast-panel-ink-target";
export const CONTRAST_PANEL_TARGET_VAR = "--contrast-panel-target";
/** Multipliers for the hand-written washes that stand in for those tokens. */
export const CONTRAST_WASH_GAIN_VAR = "--contrast-wash-gain";
export const CONTRAST_EDGE_GAIN_VAR = "--contrast-edge-gain";
export const CONTRAST_SEAM_GAIN_VAR = "--contrast-seam-gain";

const HEX_COLOR_PATTERN = /^#[0-9a-fA-F]{6}$/;

export function isHexColor(value: unknown): value is string {
  return typeof value === "string" && HEX_COLOR_PATTERN.test(value);
}

function sanitizeColor(value: unknown): string | null {
  return isHexColor(value) ? value.toLowerCase() : null;
}

function sanitizeFont(value: unknown): string | null {
  if (typeof value !== "string") return null;
  // Strip what the backend rejects (_FONT_NAME_FORBIDDEN plus control chars) so the PUT never fails;
  // also stops CSS smuggling through setProperty.
  const cleaned = value
    .replace(/[;{}()<>"'\\/,`]/g, "")
    .replace(/\p{Cc}/gu, "")
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
    const rawName = sanitizeFont(source.name);
    const name = rawName
      ? rawName.slice(0, MAX_IMPORTED_FONT_NAME_LENGTH)
      : null;
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

function isSidebarNavItemId(value: unknown): value is SidebarNavItemId {
  return SIDEBAR_NAV_ITEM_IDS.includes(value as SidebarNavItemId);
}

function sanitizeSidebarNav(value: unknown): SidebarNavItemPref[] {
  const items: SidebarNavItemPref[] = [];
  const seen = new Set<SidebarNavItemId>();
  for (const entry of Array.isArray(value) ? value : []) {
    const source = (entry ?? {}) as Partial<SidebarNavItemPref>;
    if (!isSidebarNavItemId(source.id) || seen.has(source.id)) continue;
    seen.add(source.id);
    items.push({ id: source.id, pinned: source.pinned !== false });
  }
  // Ids added after the payload was written land at the end with their default.
  for (const id of SIDEBAR_NAV_ITEM_IDS) {
    if (!seen.has(id)) items.push({ id, pinned: SIDEBAR_NAV_DEFAULT_PINNED[id] });
  }
  return items;
}

/** `undefined` predates this field; only an untouched layout gets the Projects rule. */
function sanitizeSidebarNavAuto(
  value: unknown,
  nav: SidebarNavItemPref[],
): SidebarNavItemId[] {
  if (value === undefined || value === null) {
    return isUntouchedSidebarNav(nav) ? [...SIDEBAR_NAV_AUTO_ITEM_IDS] : [];
  }
  const seen = new Set<SidebarNavItemId>();
  for (const entry of Array.isArray(value) ? value : []) {
    // Only rows that have a rule: anything else would silently pin itself.
    if (
      isSidebarNavItemId(entry) &&
      (SIDEBAR_NAV_AUTO_ITEM_IDS as readonly string[]).includes(entry)
    ) {
      seen.add(entry);
    }
  }
  return [...seen];
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
  for (const id of SIDEBAR_MENU_ITEM_IDS) {
    if (!seen.has(id))
      items.push({ id, visible: SIDEBAR_MENU_DEFAULT_VISIBLE[id] });
  }
  return items;
}

/** Malformed fields fall back to defaults so a bad payload can never wedge the UI. */
export function sanitizeCustomization(value: unknown): AppearanceCustomization {
  const source = (value ?? {}) as Partial<AppearanceCustomization> & {
    colors?: { light?: unknown; dark?: unknown };
  };
  const contrast =
    typeof source.contrast === "number" && Number.isFinite(source.contrast)
      ? Math.min(100, Math.max(0, Math.round(source.contrast)))
      : DEFAULT_CUSTOMIZATION.contrast;
  const sidebarNav = sanitizeSidebarNav(source.sidebarNav);
  return {
    colors: {
      light: sanitizeModeColors(source.colors?.light),
      dark: sanitizeModeColors(source.colors?.dark),
    },
    uiFont: sanitizeFont(source.uiFont),
    headingFont: sanitizeFont(source.headingFont),
    chatFont: sanitizeFont(source.chatFont),
    chatWidth:
      source.chatWidth === "wide" || source.chatWidth === "full"
        ? source.chatWidth
        : "standard",
    sentAttachments:
      source.sentAttachments === "list" || source.sentAttachments === "chips"
        ? source.sentAttachments
        : "auto",
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
    sidebarMenu: sanitizeSidebarMenu(source.sidebarMenu),
    sidebarNav,
    sidebarNavAuto: sanitizeSidebarNavAuto(source.sidebarNavAuto, sidebarNav),
  };
}

function isUntouchedSidebarNav(nav: SidebarNavItemPref[]): boolean {
  const stored = JSON.stringify(nav);
  if (stored === JSON.stringify(DEFAULT_CUSTOMIZATION.sidebarNav)) return true;
  // Sanitize each layout too, since the stored one has gained later ids.
  return SHIPPED_SIDEBAR_NAV_DEFAULTS.some(
    (layout) => JSON.stringify(sanitizeSidebarNav(layout)) === stored,
  );
}

export function migrateShippedSidebarNavDefault(
  customization: AppearanceCustomization,
  storedVersion: number,
  migrationVersion: number,
): AppearanceCustomization {
  // Once persisted at this version, the same layout may be a deliberate choice.
  if (storedVersion >= migrationVersion) return customization;
  return isUntouchedSidebarNav(customization.sidebarNav)
    ? {
        ...customization,
        sidebarNav: [...DEFAULT_CUSTOMIZATION.sidebarNav],
      }
    : customization;
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
      version: 8,
      storage: createJSONStorage(() => guardedLocalStorage),
      migrate: (persisted, version) => {
        const state = (persisted ?? {}) as Partial<AppearanceCustomState>;
        const customization = migrateShippedSidebarNavDefault(
          sanitizeCustomization(state.customization),
          version,
          8,
        );
        return { customization } as AppearanceCustomState;
      },
      // Sanitize on every rehydrate: a same-version payload from an older bundle can miss fields.
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

const DEFAULT_SANS_STACK =
  '"Inter Variable", ui-sans-serif, sans-serif, system-ui';
const DEFAULT_HEADING_STACK =
  '"Hellix", "Space Grotesk Variable", var(--font-sans)';
const DEFAULT_MONO_STACK = "JetBrains Mono, monospace";

function hexLuminance(hex: string): number {
  const channel = (i: number) => {
    const c = Number.parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
}

const FOREGROUND_DARK = "#111417";
const FOREGROUND_LIGHT = "#ffffff";
const FOREGROUND_DARK_FALLBACK = "#000000";
const FOREGROUND_CONTRAST_FLOOR = 4.5;
const ACCENT_TEXT_FLOOR = 2.5;
const ACCENT_TEXT_WASH_OPACITY = 0.2;
// Full 8-bit range so a narrow valid contrast band is not skipped.
const MIX_STEPS = 255;

function contrastRatio(a: number, b: number): number {
  const [high, low] = a >= b ? [a, b] : [b, a];
  return (high + 0.05) / (low + 0.05);
}

function mixColors(hex: string, target: string, amount: number): string {
  const channel = (index: number) => {
    const value = Number.parseInt(hex.slice(index, index + 2), 16);
    const targetValue = Number.parseInt(target.slice(index, index + 2), 16);
    return Math.round(value + (targetValue - value) * amount)
      .toString(16)
      .padStart(2, "0");
  };
  return `#${channel(1)}${channel(3)}${channel(5)}`;
}

/** Compare both ratios: a fixed luminance threshold put white on mid-tone accents. */
function readableForeground(hex: string): string {
  const accent = hexLuminance(hex);
  const darkContrast = contrastRatio(accent, hexLuminance(FOREGROUND_DARK));
  const lightContrast = contrastRatio(accent, hexLuminance(FOREGROUND_LIGHT));
  const preferred =
    darkContrast >= lightContrast ? FOREGROUND_DARK : FOREGROUND_LIGHT;
  if (Math.max(darkContrast, lightContrast) >= FOREGROUND_CONTRAST_FLOOR) {
    return preferred;
  }
  // Both preferred inks can fall below AA in a narrow band; true black closes it.
  return FOREGROUND_DARK_FALLBACK;
}

type PaletteSurfaces = { background: string; elevated: readonly string[] };

const PALETTE_SURFACES: Record<ResolvedTheme, PaletteSurfaces> = {
  light: { background: "#ffffff", elevated: ["#ffffff"] },
  dark: { background: "#181818", elevated: ["#272727"] },
};

function surfacesFor(
  palette: ColorThemeId,
  resolved: ResolvedTheme,
): PaletteSurfaces {
  const { background, surface } = COLOR_THEMES[palette][resolved];
  if (!surface) return PALETTE_SURFACES[resolved];
  return {
    background,
    elevated: resolved === "light" ? [surface, "#ffffff"] : [surface],
  };
}

/** Black for ink darker than its page, white for lighter, so raising never heads into the page. */
function inkPole(ink: string, page: string): string {
  return hexLuminance(ink) <= hexLuminance(page) ? "#000000" : "#ffffff";
}

function minimumAccentTextContrast(
  accent: string,
  backgrounds: readonly string[],
): number {
  const accentLuminance = hexLuminance(accent);
  const against = (background: string) => {
    const wash = mixColors(background, accent, ACCENT_TEXT_WASH_OPACITY);
    return Math.min(
      contrastRatio(accentLuminance, hexLuminance(background)),
      contrastRatio(accentLuminance, hexLuminance(wash)),
    );
  };
  return Math.min(...backgrounds.map(against));
}

function minimumPlainTextContrast(
  accent: string,
  backgrounds: readonly string[],
): number {
  const accentLuminance = hexLuminance(accent);
  return Math.min(
    ...backgrounds.map((background) =>
      contrastRatio(accentLuminance, hexLuminance(background)),
    ),
  );
}

function findAccentCorrection(
  accent: string,
  backgrounds: readonly string[],
  score: (hex: string, surfaces: readonly string[]) => number,
): string | null {
  const clears = (hex: string) => score(hex, backgrounds) >= ACCENT_TEXT_FLOOR;
  if (clears(accent)) {
    return accent;
  }

  const targets = [FOREGROUND_DARK_FALLBACK, FOREGROUND_LIGHT].sort(
    (a, b) => score(b, backgrounds) - score(a, backgrounds),
  );
  for (let step = 1; step <= MIX_STEPS; step += 1) {
    for (const target of targets) {
      const candidate = mixColors(accent, target, step / MIX_STEPS);
      if (clears(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}

/**
 * text-primary paints the accent on surfaces and washes up to 20%, so hold custom accents to 2.5:1
 * with the smallest correction in either direction; if washes are impossible, keep the plain floor.
 */
function legibleAccent(accent: string, backgrounds: readonly string[]): string {
  const washSafe = findAccentCorrection(
    accent,
    backgrounds,
    minimumAccentTextContrast,
  );
  if (washSafe) {
    return washSafe;
  }

  const plainSafe = findAccentCorrection(
    accent,
    backgrounds,
    minimumPlainTextContrast,
  );
  if (plainSafe) {
    return plainSafe;
  }

  const targets = [FOREGROUND_DARK_FALLBACK, FOREGROUND_LIGHT].sort(
    (a, b) =>
      minimumPlainTextContrast(b, backgrounds) -
      minimumPlainTextContrast(a, backgrounds),
  );
  return targets[0] ?? FOREGROUND_DARK_FALLBACK;
}

/** dataUrl is tracked so a same-name re-import replaces the face. */
const registeredFontFaces = new Map<
  string,
  { face: FontFace; dataUrl: string }
>();

function syncImportedFonts(fonts: ImportedFont[]): void {
  if (typeof document === "undefined" || !("fonts" in document)) return;
  // A stale persisted payload without importedFonts must not crash the applier.
  const wanted = new Map(
    (Array.isArray(fonts) ? fonts : []).map((f) => [f.name, f.dataUrl]),
  );
  // document.fonts is a set of FontFace objects, not keyed by family, so delete stale faces first.
  for (const [name, entry] of registeredFontFaces) {
    if (wanted.get(name) !== entry.dataUrl) {
      document.fonts.delete(entry.face);
      registeredFontFaces.delete(name);
    }
  }
  for (const [name, dataUrl] of wanted) {
    if (registeredFontFaces.has(name)) continue;
    try {
      const face = new FontFace(name, `url(${dataUrl})`);
      registeredFontFaces.set(name, { face, dataUrl });
      document.fonts.add(face);
      face.load().catch(() => {
        document.fonts.delete(face);
        // A same-name re-import may have replaced it while this load was pending.
        if (registeredFontFaces.get(name)?.face === face) {
          registeredFontFaces.delete(name);
        }
      });
    } catch {
      registeredFontFaces.delete(name);
    }
  }
}

/** Set only while an accent is picked. --ring and --verified are deliberately not included. */
const ACCENT_VARS = ["--control-accent", "--chart-1", "--primary"] as const;
const ACCENT_FG_VARS = [
  "--control-accent-foreground",
  "--primary-foreground",
] as const;

/** Inline vars beat palette blocks; the default customization leaves <html> untouched. */
export function applyCustomizationToDocument(
  c: AppearanceCustomization,
  resolved: ResolvedTheme,
  palette: ColorThemeId = "standard",
): void {
  if (typeof document === "undefined") return;
  const el = document.documentElement;
  const style = el.style;

  const setVar = (name: string, value: string | null) => {
    if (value === null) style.removeProperty(name);
    else style.setProperty(name, value);
  };

  const colors = c.colors[resolved];
  const paletteSurfaces = surfacesFor(palette, resolved);

  const accent = colors.accent
    ? legibleAccent(colors.accent, [
        colors.background ?? paletteSurfaces.background,
        ...paletteSurfaces.elevated,
      ])
    : null;
  for (const name of ACCENT_VARS) setVar(name, accent);
  for (const name of ACCENT_FG_VARS) {
    // Keyed off the corrected accent, since that is the color labels sit on.
    setVar(name, accent ? readableForeground(accent) : null);
  }
  setVar("--background", colors.background);
  // Written as the base so a custom foreground still follows the contrast curve.
  setVar("--foreground-base", colors.foreground);
  // Clear the token older builds wrote inline; it would pin the ink.
  setVar("--foreground", null);
  // keep the full-width inset from making narrow panes smaller than wide.
  setVar(
    "--custom-chat-max-width",
    c.chatWidth === "full"
      ? "max(72rem, calc(100% - 6rem))"
      : c.chatWidth === "wide"
        ? "72rem"
        : null,
  );
  // Full width's cap is a percentage, so give the shell the same number from its capped parent.
  setVar(
    "--custom-chat-shell-max-width",
    c.chatWidth === "full"
      ? "min(100%, max(calc(72rem - 1.5rem), calc(100% - 1.5rem)))"
      : null,
  );

  syncImportedFonts(c.importedFonts);

  // sanitizeFont strips quotes, so quoting here is always safe.
  setVar(
    "--font-sans",
    c.uiFont ? `"${c.uiFont}", ${DEFAULT_SANS_STACK}` : null,
  );
  // Custom interface fonts cascade into chat and opt out of its Inter tuning.
  el.toggleAttribute("data-ui-font", Boolean(c.uiFont));
  setVar(
    "--font-heading",
    c.headingFont ? `"${c.headingFont}", ${DEFAULT_HEADING_STACK}` : null,
  );
  // Only while chosen, so elements pinned to their own default (chat greeting) can follow a pick.
  setVar(
    "--custom-heading-font",
    c.headingFont ? `"${c.headingFont}", ${DEFAULT_HEADING_STACK}` : null,
  );
  setVar(
    "--font-mono",
    c.codeFont ? `"${c.codeFont}", ${DEFAULT_MONO_STACK}` : null,
  );
  // Chat code defaults to Fira Code, not --font-mono, so it needs a dedicated token.
  setVar(
    "--custom-code-font",
    c.codeFont ? `"${c.codeFont}", "Fira Code", ui-monospace, monospace` : null,
  );

  if (c.chatFont) {
    el.setAttribute("data-chat-font", "");
    setVar("--custom-chat-font", `"${c.chatFont}", ${DEFAULT_SANS_STACK}`);
  } else {
    el.removeAttribute("data-chat-font");
    setVar("--custom-chat-font", null);
  }

  // Scales typography tokens, never the root font size, so rem-based layout does not move.
  const effectiveUiFontSize = c.uiFontSize ?? UI_FONT_SIZE_RANGE.default;
  if (effectiveUiFontSize !== UI_FONT_SIZE_RANGE.default) {
    setVar(
      "--ui-font-size-scale",
      String(effectiveUiFontSize / UI_FONT_SIZE_CSS_BASE),
    );
    el.setAttribute("data-ui-font-size", String(effectiveUiFontSize));
  } else {
    setVar("--ui-font-size-scale", null);
    el.removeAttribute("data-ui-font-size");
  }
  // index.css derives --ui-font-scale; clear the value older builds wrote.
  setVar("--ui-font-scale", null);
  // Older builds scaled the root font size; clear any stale inline value.
  style.removeProperty("font-size");

  if (c.codeFontSize !== null) {
    el.setAttribute("data-code-font-size", "");
    setVar("--custom-code-font-size", `${c.codeFontSize}px`);
  } else {
    el.removeAttribute("data-code-font-size");
    setVar("--custom-code-font-size", null);
  }

  if (c.contrast !== 50) {
    // Everything mixes toward --contrast-target: the foreground above 50, the background below.
    const distance = Math.abs(c.contrast - 50) / 50;
    const raising = c.contrast > 50;
    const mix = (ceiling: number) => `${Math.round(distance * ceiling)}%`;
    el.setAttribute("data-contrast-adjust", "");
    setVar(
      "--contrast-target",
      raising ? "var(--foreground)" : "var(--background)",
    );
    // Asymmetric: lowering flattens surfaces, raising only nudges so cards do not read as blocks.
    setVar(CONTRAST_SURFACE_MIX_VAR, mix(raising ? 4 : 70));
    setVar(CONTRAST_FILL_MIX_VAR, mix(raising ? 10 : 62));
    setVar(CONTRAST_LINE_MIX_VAR, mix(raising ? 45 : 80));
    setVar(CONTRAST_CONTROL_MIX_VAR, mix(raising ? 45 : 55));
    setVar(CONTRAST_STATE_MIX_VAR, mix(raising ? 16 : 55));
    setVar(CONTRAST_TEXT_MIX_VAR, mix(raising ? 40 : 30));
    // Stops well short so body copy stays readable at 0.
    const palettePole = resolved === "light" ? "#000000" : "#ffffff";
    // Palette surfaces keep their colors under a custom foreground, so they raise toward the pole.
    setVar(
      CONTRAST_PANEL_TARGET_VAR,
      raising && colors.foreground ? palettePole : null,
    );
    setVar(
      CONTRAST_INK_TARGET_VAR,
      raising
        ? colors.foreground
          ? inkPole(
              colors.foreground,
              colors.background ?? paletteSurfaces.background,
            )
          : palettePole
        : "var(--background)",
    );
    setVar(
      CONTRAST_PANEL_INK_TARGET_VAR,
      raising ? palettePole : "var(--background)",
    );
    setVar(CONTRAST_INK_MIX_VAR, mix(raising ? 70 : 30));
    const gain = (span: number) =>
      (raising ? 1 + distance * span : 1 - distance * span).toFixed(3);
    setVar(CONTRAST_WASH_GAIN_VAR, gain(raising ? 1 : 0.4));
    setVar(CONTRAST_EDGE_GAIN_VAR, gain(raising ? 0.9 : 0.8));
    setVar(
      CONTRAST_SEAM_GAIN_VAR,
      (1 - distance * (raising ? 0.9 : 0.15)).toFixed(3),
    );
  } else {
    el.removeAttribute("data-contrast-adjust");
    setVar("--contrast-target", null);
    setVar(CONTRAST_SURFACE_MIX_VAR, null);
    setVar(CONTRAST_FILL_MIX_VAR, null);
    setVar(CONTRAST_LINE_MIX_VAR, null);
    setVar(CONTRAST_CONTROL_MIX_VAR, null);
    setVar(CONTRAST_STATE_MIX_VAR, null);
    setVar(CONTRAST_TEXT_MIX_VAR, null);
    setVar(CONTRAST_INK_TARGET_VAR, null);
    setVar(CONTRAST_PANEL_INK_TARGET_VAR, null);
    setVar(CONTRAST_PANEL_TARGET_VAR, null);
    setVar(CONTRAST_INK_MIX_VAR, null);
    setVar(CONTRAST_WASH_GAIN_VAR, null);
    setVar(CONTRAST_EDGE_GAIN_VAR, null);
    setVar(CONTRAST_SEAM_GAIN_VAR, null);
  }

  el.classList.toggle("pointer-cursors", c.pointerCursors);
  el.classList.toggle("force-reduced-motion", c.reduceMotion === "on");
  // "off" overrides OS reduced motion; index.css media rules skip html.force-motion.
  el.classList.toggle("force-motion", c.reduceMotion === "off");
  el.classList.toggle("no-font-smoothing", !c.fontSmoothing);
}

/** For imperative motion (confetti, view transitions) CSS/MotionConfig cannot reach. */
export function prefersReducedMotion(): boolean {
  const setting =
    useAppearanceCustomStore.getState().customization.reduceMotion;
  if (setting === "on") return true;
  if (setting === "off") return false;
  return (
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true
  );
}
