// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
    appearance: true,
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
  // Last and under "More", so it joins every saved layout without moving anything.
  "tasks",
] as const;

export type SidebarNavItemId = (typeof SIDEBAR_NAV_ITEM_IDS)[number];

export type SidebarNavItemPref = {
  id: SidebarNavItemId;
  pinned: boolean;
};

export const SIDEBAR_NAV_AUTO_ITEM_IDS = [
  "projects",
] as const satisfies readonly SidebarNavItemId[];

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
  tasks: false,
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
  /** 0–100; null = the background image's own outline, or none without one. */
  composerOutline: number | null;
  /** 0–100; null = each theme's own borders. */
  elementOutline: number | null;
  pointerCursors: boolean;
  reduceMotion: ReduceMotionSetting;
  /** true = the app default (antialiased). */
  fontSmoothing: boolean;
  codeLineNumbers: boolean;
  codeWordWrap: boolean;
  codeLigatures: boolean;
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
  composerOutline: null,
  elementOutline: null,
  pointerCursors: false,
  reduceMotion: "on",
  fontSmoothing: true,
  codeLineNumbers: false,
  codeWordWrap: false,
  codeLigatures: true,
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

const HEX_COLOR_PATTERN = /^#[0-9a-fA-F]{6}$/;

export function isHexColor(value: unknown): value is string {
  return typeof value === "string" && HEX_COLOR_PATTERN.test(value);
}

export function sanitizeColor(value: unknown): string | null {
  return isHexColor(value) ? value.toLowerCase() : null;
}

export type CustomizationIssueKind = "invalid" | "clamped" | "dropped";

export type CustomizationIssue = {
  /** Position in the stored layout, e.g. "colors.light.accent" or "importedFonts[1]". */
  path: string;
  kind: CustomizationIssueKind;
};

type ReportIssue = (path: string, kind: CustomizationIssueKind) => void;

const ignoreIssue: ReportIssue = () => {};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function checkedColor(
  value: unknown,
  path: string,
  report: ReportIssue,
): string | null {
  if (value != null && !isHexColor(value)) report(path, "invalid");
  return sanitizeColor(value);
}

export function sanitizeFontName(value: unknown): string | null {
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

/** The family a font file registers under, e.g. "Brand_Sans-Bold.woff2" -> "Brand Sans Bold". */
export function fontFamilyFromFileName(fileName: string): string {
  const base = fileName.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
  return sanitizeFontName(base) ?? "";
}

function checkedFont(
  value: unknown,
  path: string,
  report: ReportIssue,
): string | null {
  const cleaned = sanitizeFontName(value);
  if (value != null && cleaned !== value) report(path, "invalid");
  return cleaned;
}

/** Rounds and clamps. A finite number that had to move is "clamped"; anything else is "invalid". */
function sanitizeSize(
  value: unknown,
  range: { min: number; max: number },
  path: string,
  report: ReportIssue,
): number | null {
  if (value == null) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    report(path, "invalid");
    return null;
  }
  const next = Math.min(range.max, Math.max(range.min, Math.round(value)));
  if (next !== value) report(path, "clamped");
  return next;
}

function readBoolean(
  value: unknown,
  path: string,
  report: ReportIssue,
): boolean | undefined {
  if (typeof value === "boolean") return value;
  if (value != null) report(path, "invalid");
  return undefined;
}

function readChoice<T extends string>(
  value: unknown,
  choices: readonly T[],
  fallback: T,
  path: string,
  report: ReportIssue,
): T {
  if (value == null) return fallback;
  if (choices.includes(value as T)) return value as T;
  report(path, "invalid");
  return fallback;
}

function sanitizeModeColors(
  value: unknown,
  path: string,
  report: ReportIssue,
): CustomModeColors {
  if (value != null && !isRecord(value)) report(path, "invalid");
  const source = (value ?? {}) as Partial<CustomModeColors>;
  return {
    accent: checkedColor(source.accent, `${path}.accent`, report),
    background: checkedColor(source.background, `${path}.background`, report),
    foreground: checkedColor(source.foreground, `${path}.foreground`, report),
  };
}

const FONT_DATA_URL_PATTERN =
  /^data:(?:font\/(?:woff2?|ttf|otf|sfnt)|application\/(?:octet-stream|x-font-\w+|font-\w+));base64,[A-Za-z0-9+/=]+$/;

function sanitizeImportedFonts(
  value: unknown,
  report: ReportIssue,
): ImportedFont[] {
  if (!Array.isArray(value)) {
    if (value != null) report("importedFonts", "invalid");
    return [];
  }
  const fonts: ImportedFont[] = [];
  const seen = new Set<string>();
  let total = 0;
  for (const [index, entry] of value.entries()) {
    const path = `importedFonts[${index}]`;
    // Stop at the cap, like before: one issue stands for everything after it.
    if (fonts.length >= MAX_IMPORTED_FONTS) {
      report(path, "dropped");
      break;
    }
    const source = (entry ?? {}) as Partial<ImportedFont>;
    const rawName = sanitizeFontName(source.name);
    const name = rawName
      ? rawName.slice(0, MAX_IMPORTED_FONT_NAME_LENGTH)
      : null;
    const dataUrl = source.dataUrl;
    if (
      !name ||
      seen.has(name) ||
      typeof dataUrl !== "string" ||
      dataUrl.length > MAX_IMPORTED_FONT_DATA_URL_LENGTH ||
      total + dataUrl.length > MAX_TOTAL_IMPORTED_FONT_DATA_URL_LENGTH ||
      !FONT_DATA_URL_PATTERN.test(dataUrl)
    ) {
      report(path, "dropped");
      continue;
    }
    if (name !== source.name) report(`${path}.name`, "invalid");
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

function sanitizeSidebarNav(
  value: unknown,
  report: ReportIssue = ignoreIssue,
): SidebarNavItemPref[] {
  if (value != null && !Array.isArray(value)) report("sidebarNav", "invalid");
  const items: SidebarNavItemPref[] = [];
  const seen = new Set<SidebarNavItemId>();
  (Array.isArray(value) ? value : []).forEach((entry, index) => {
    const source = (entry ?? {}) as Partial<SidebarNavItemPref>;
    if (!isSidebarNavItemId(source.id) || seen.has(source.id)) {
      report(`sidebarNav[${index}]`, "dropped");
      return;
    }
    seen.add(source.id);
    readBoolean(source.pinned, `sidebarNav[${index}].pinned`, report);
    items.push({ id: source.id, pinned: source.pinned !== false });
  });
  // Ids added after the payload was written land at the end with their default.
  for (const id of SIDEBAR_NAV_ITEM_IDS) {
    if (!seen.has(id))
      items.push({ id, pinned: SIDEBAR_NAV_DEFAULT_PINNED[id] });
  }
  return items;
}

/** `undefined` predates this field; only an untouched layout gets the Projects rule. */
function sanitizeSidebarNavAuto(
  value: unknown,
  nav: SidebarNavItemPref[],
  report: ReportIssue = ignoreIssue,
): SidebarNavItemId[] {
  if (value === undefined || value === null) {
    return isUntouchedSidebarNav(nav) ? [...SIDEBAR_NAV_AUTO_ITEM_IDS] : [];
  }
  if (!Array.isArray(value)) report("sidebarNavAuto", "invalid");
  const seen = new Set<SidebarNavItemId>();
  (Array.isArray(value) ? value : []).forEach((entry, index) => {
    // Only rows that have a rule: anything else would silently pin itself.
    if (
      isSidebarNavItemId(entry) &&
      (SIDEBAR_NAV_AUTO_ITEM_IDS as readonly string[]).includes(entry) &&
      !seen.has(entry)
    ) {
      seen.add(entry);
    } else {
      report(`sidebarNavAuto[${index}]`, "dropped");
    }
  });
  return [...seen];
}

function sanitizeSidebarMenu(
  value: unknown,
  report: ReportIssue,
): SidebarMenuItemPref[] {
  if (value != null && !Array.isArray(value)) report("sidebarMenu", "invalid");
  const items: SidebarMenuItemPref[] = [];
  const seen = new Set<SidebarMenuItemId>();
  (Array.isArray(value) ? value : []).forEach((entry, index) => {
    const source = (entry ?? {}) as Partial<SidebarMenuItemPref>;
    if (!isSidebarMenuItemId(source.id) || seen.has(source.id)) {
      report(`sidebarMenu[${index}]`, "dropped");
      return;
    }
    seen.add(source.id);
    readBoolean(source.visible, `sidebarMenu[${index}].visible`, report);
    items.push({ id: source.id, visible: source.visible !== false });
  });
  for (const id of SIDEBAR_MENU_ITEM_IDS) {
    if (!seen.has(id))
      items.push({ id, visible: SIDEBAR_MENU_DEFAULT_VISIBLE[id] });
  }
  return items;
}

/**
 * Malformed fields fall back to defaults so a bad payload can never wedge the UI. Every value that
 * was not kept as written is reported. A missing or null field is the default and is never reported.
 * The customization returned is the one sanitizeCustomization has always produced.
 */
export const MAX_CUSTOMIZATION_ISSUES = 100;

export function sanitizeCustomizationWithIssues(value: unknown): {
  value: AppearanceCustomization;
  issues: CustomizationIssue[];
  /** Issues past MAX_CUSTOMIZATION_ISSUES, counted but not kept. */
  omittedIssues: number;
} {
  const issues: CustomizationIssue[] = [];
  let omittedIssues = 0;
  // A hostile payload can hold millions of bad rows; keep the first few and count the rest.
  const report: ReportIssue = (path, kind) => {
    if (issues.length < MAX_CUSTOMIZATION_ISSUES) issues.push({ path, kind });
    else omittedIssues += 1;
  };
  const source = (
    isRecord(value) ? value : {}
  ) as Partial<AppearanceCustomization> & {
    colors?: { light?: unknown; dark?: unknown };
  };
  if (source.colors != null && !isRecord(source.colors)) {
    report("colors", "invalid");
  }
  const contrast =
    sanitizeSize(source.contrast, { min: 0, max: 100 }, "contrast", report) ??
    DEFAULT_CUSTOMIZATION.contrast;
  const sidebarNav = sanitizeSidebarNav(source.sidebarNav, report);
  const customization: AppearanceCustomization = {
    colors: {
      light: sanitizeModeColors(source.colors?.light, "colors.light", report),
      dark: sanitizeModeColors(source.colors?.dark, "colors.dark", report),
    },
    uiFont: checkedFont(source.uiFont, "uiFont", report),
    headingFont: checkedFont(source.headingFont, "headingFont", report),
    chatFont: checkedFont(source.chatFont, "chatFont", report),
    chatWidth: readChoice(
      source.chatWidth,
      ["standard", "wide", "full"],
      "standard",
      "chatWidth",
      report,
    ),
    sentAttachments: readChoice(
      source.sentAttachments,
      ["auto", "list", "chips"],
      "auto",
      "sentAttachments",
      report,
    ),
    codeFont: checkedFont(source.codeFont, "codeFont", report),
    importedFonts: sanitizeImportedFonts(source.importedFonts, report),
    uiFontSize: sanitizeSize(
      source.uiFontSize,
      UI_FONT_SIZE_RANGE,
      "uiFontSize",
      report,
    ),
    codeFontSize: sanitizeSize(
      source.codeFontSize,
      CODE_FONT_SIZE_RANGE,
      "codeFontSize",
      report,
    ),
    contrast,
    composerOutline: sanitizeSize(
      source.composerOutline,
      { min: 0, max: 100 },
      "composerOutline",
      report,
    ),
    elementOutline: sanitizeSize(
      source.elementOutline,
      { min: 0, max: 100 },
      "elementOutline",
      report,
    ),
    pointerCursors:
      readBoolean(source.pointerCursors, "pointerCursors", report) === true,
    reduceMotion: readChoice(
      source.reduceMotion,
      ["system", "on", "off"],
      "on",
      "reduceMotion",
      report,
    ),
    fontSmoothing:
      readBoolean(source.fontSmoothing, "fontSmoothing", report) !== false,
    codeLineNumbers:
      readBoolean(source.codeLineNumbers, "codeLineNumbers", report) === true,
    codeWordWrap:
      readBoolean(source.codeWordWrap, "codeWordWrap", report) === true,
    codeLigatures:
      readBoolean(source.codeLigatures, "codeLigatures", report) !== false,
    sidebarMenu: sanitizeSidebarMenu(source.sidebarMenu, report),
    sidebarNav,
    sidebarNavAuto: sanitizeSidebarNavAuto(
      source.sidebarNavAuto,
      sidebarNav,
      report,
    ),
  };
  return { value: customization, issues, omittedIssues };
}

export function sanitizeCustomization(value: unknown): AppearanceCustomization {
  return sanitizeCustomizationWithIssues(value).value;
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
  // Once persisted at this version, never adopt the same layout again; it may be a deliberate choice.
  if (storedVersion >= migrationVersion) return customization;
  return isUntouchedSidebarNav(customization.sidebarNav)
    ? {
        ...customization,
        sidebarNav: [...DEFAULT_CUSTOMIZATION.sidebarNav],
      }
    : customization;
}

/** Older records adopt the Appearance menu item once; after that a hidden one stays hidden. */
export function migrateAppearanceMenuDefault(
  customization: AppearanceCustomization,
  storedVersion: number,
  migrationVersion: number,
): AppearanceCustomization {
  if (storedVersion >= migrationVersion) return customization;
  return {
    ...customization,
    sidebarMenu: customization.sidebarMenu.map((item) =>
      item.id === "appearance" ? { ...item, visible: true } : item,
    ),
  };
}

export function isDefaultCustomization(c: AppearanceCustomization): boolean {
  return JSON.stringify(c) === JSON.stringify(DEFAULT_CUSTOMIZATION);
}
