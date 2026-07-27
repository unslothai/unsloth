// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { type TranslationKey, useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Folder01Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion, useReducedMotionConfig } from "motion/react";
import {
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import {
  CODE_FONT_SIZE_RANGE,
  type CustomModeColors,
  DEFAULT_CUSTOMIZATION,
  MAX_IMPORTED_FONTS,
  MAX_TOTAL_IMPORTED_FONT_DATA_URL_LENGTH,
  type ReduceMotionSetting,
  UI_FONT_SIZE_RANGE,
  isDefaultCustomization,
  useAppearanceCustomStore,
} from "../stores/appearance-custom-store";
import {
  type Palette,
  type ResolvedTheme,
  usePalette,
  useTheme,
} from "../stores/theme-store";
import { ColorPickerSwatch } from "./color-picker";

/* ------------------------------- Colors -------------------------------- */

// Seed values shown in the pickers while no override is set. Mirrors the
// palette token values in index.css (foregrounds converted from oklch).
type DefaultModeColors = { [K in keyof CustomModeColors]: string };
const PALETTE_DEFAULT_COLORS: Record<
  Palette,
  Record<ResolvedTheme, DefaultModeColors>
> = {
  standard: {
    light: { accent: "#17b88b", background: "#fefefd", foreground: "#262626" },
    dark: { accent: "#17b88b", background: "#181818", foreground: "#ececec" },
  },
  classic: {
    light: { accent: "#339cff", background: "#ffffff", foreground: "#1a1c1f" },
    dark: { accent: "#4dabff", background: "#181818", foreground: "#ececec" },
  },
  minimal: {
    light: { accent: "#171717", background: "#ffffff", foreground: "#171717" },
    dark: { accent: "#ededed", background: "#181818", foreground: "#ededed" },
  },
};

/**
 * Color override control for the CURRENTLY ACTIVE resolved mode. Only the
 * active mode is editable; the other mode's overrides stay stored and take
 * effect when the color scheme flips.
 */
export function ActiveColorControl({
  colorKey,
  label,
}: {
  colorKey: keyof CustomModeColors;
  label: string;
}) {
  const t = useT();
  const { resolved } = useTheme();
  const { palette } = usePalette();
  const override = useAppearanceCustomStore(
    (s) => s.customization.colors[resolved][colorKey],
  );
  const setColor = useAppearanceCustomStore((s) => s.setColor);
  const fallback = PALETTE_DEFAULT_COLORS[palette][resolved][colorKey];
  const value = override ?? fallback;
  return (
    <div className="flex items-center gap-2">
      {override !== null && (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-xs text-muted-foreground"
          onClick={() => setColor(resolved, colorKey, null)}
        >
          {t("settings.appearance.custom.reset")}
        </Button>
      )}
      <ColorPickerSwatch
        value={value}
        onChange={(hex) => setColor(resolved, colorKey, hex)}
        label={label}
      />
    </div>
  );
}

/* ------------------------------ Typography ------------------------------ */

/** Font each slot resolves to when no override is set (see index.css). */
const DEFAULT_FONT_NAMES = {
  ui: "Inter Variable",
  heading: "Hellix",
  chat: "Inter Variable",
  code: "JetBrains Mono",
} as const;

/** Fonts Unsloth Studio already ships (bundled @font-face / fontsource). */
const BUNDLED_FONTS = [
  "Inter Variable",
  "Hellix",
  "Space Grotesk Variable",
  "Figtree Variable",
  "JetBrains Mono",
  "Fira Code",
] as const;

/* ----------------------------- Device fonts ------------------------------ */

// Fallback candidates probed by canvas measurement when the Local Font
// Access API (Chromium-only) is unavailable or denied.
const CANDIDATE_DEVICE_FONTS = [
  "American Typewriter",
  "Andale Mono",
  "Arial",
  "Avenir",
  "Avenir Next",
  "Baskerville",
  "Calibri",
  "Cambria",
  "Candara",
  "Cantarell",
  "Charter",
  "Comic Sans MS",
  "Consolas",
  "Constantia",
  "Corbel",
  "Courier",
  "Courier New",
  "DejaVu Sans",
  "DejaVu Sans Mono",
  "DejaVu Serif",
  "Didot",
  "Fira Sans",
  "Franklin Gothic Medium",
  "Futura",
  "Geneva",
  "Georgia",
  "Gill Sans",
  "Helvetica",
  "Helvetica Neue",
  "Hoefler Text",
  "IBM Plex Mono",
  "IBM Plex Sans",
  "Impact",
  "Inconsolata",
  "Iosevka",
  "Lato",
  "Liberation Mono",
  "Liberation Sans",
  "Liberation Serif",
  "Lucida Grande",
  "Menlo",
  "Monaco",
  "Montserrat",
  "Noto Sans",
  "Noto Serif",
  "Nunito",
  "Open Sans",
  "Optima",
  "Palatino",
  "Roboto",
  "Rockwell",
  "Segoe UI",
  "Seravek",
  "Source Sans Pro",
  "Tahoma",
  "Times",
  "Times New Roman",
  "Trebuchet MS",
  "Ubuntu",
  "Verdana",
];

function detectFontsByMeasurement(candidates: string[]): string[] {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  const sample = "mmmmmmmmmwwwwwwwlli";
  const baselines = ["monospace", "sans-serif", "serif"].map((base) => {
    ctx.font = `72px ${base}`;
    return { base, width: ctx.measureText(sample).width };
  });
  return candidates.filter((family) =>
    baselines.some(({ base, width }) => {
      ctx.font = `72px "${family}", ${base}`;
      return ctx.measureText(sample).width !== width;
    }),
  );
}

let deviceFontsCache: string[] | null = null;

async function loadDeviceFonts(): Promise<string[]> {
  if (deviceFontsCache) return deviceFontsCache;
  let families: string[] = [];
  const query = (
    globalThis as {
      queryLocalFonts?: () => Promise<{ family: string }[]>;
    }
  ).queryLocalFonts;
  if (typeof query === "function") {
    try {
      const fonts = await query();
      families = [...new Set(fonts.map((f) => f.family))];
    } catch {
      // permission denied; fall back to measurement probing
    }
  }
  if (families.length === 0) {
    families = detectFontsByMeasurement(CANDIDATE_DEVICE_FONTS);
  }
  deviceFontsCache = families.sort((a, b) => a.localeCompare(b));
  return deviceFontsCache;
}

function FontSelect({
  value,
  defaultFont,
  onCommit,
  ariaLabel,
}: {
  value: string | null;
  defaultFont: string;
  onCommit: (next: string | null) => void;
  ariaLabel: string;
}) {
  const t = useT();
  const defaultLabel = `${defaultFont} (${t("settings.appearance.custom.fontDefault")})`;
  const [open, setOpen] = useState(false);
  const [deviceFonts, setDeviceFonts] = useState<string[] | null>(
    deviceFontsCache,
  );
  const importedFonts = useAppearanceCustomStore(
    (s) => s.customization.importedFonts,
  );
  const removeImportedFont = useAppearanceCustomStore(
    (s) => s.removeImportedFont,
  );

  const handleOpenChange = (next: boolean) => {
    setOpen(next);
    // Kick off inside the click gesture: queryLocalFonts may need transient
    // user activation for its permission prompt.
    if (next && deviceFonts === null) {
      void loadDeviceFonts().then(setDeviceFonts);
    }
  };

  const select = (next: string | null) => {
    onCommit(next);
    setOpen(false);
  };

  const folderFonts = useFolderFonts();
  const { inputs: uploadInputs, requestUpload, requestFolder, importFile } =
    useFontImport((name) => select(name));

  const knownNames = new Set<string>([
    ...BUNDLED_FONTS,
    ...importedFonts.map((f) => f.name),
  ]);
  const deviceOnlyFonts = (deviceFonts ?? []).filter((f) => !knownNames.has(f));

  const renderItem = (font: string) => (
    <CommandItem
      key={font}
      value={font}
      onSelect={() => select(font)}
      data-checked={value === font}
      className="cursor-pointer rounded-[11px]"
    >
      {/* Preview each entry in its own typeface. */}
      <span
        className="min-w-0 truncate"
        style={{ fontFamily: `"${font}", var(--font-sans)` }}
      >
        {font}
      </span>
    </CommandItem>
  );

  return (
    <Popover open={open} onOpenChange={handleOpenChange} modal={true}>
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          aria-label={ariaLabel}
          aria-expanded={open}
          className="flex h-8 w-48 cursor-pointer items-center justify-between gap-1.5 rounded-full border border-border bg-background px-3.5 text-xs outline-none transition-colors hover:bg-accent/50 focus-visible:border-ring dark:focus-visible:border-transparent dark:focus-visible:bg-white/[0.12] dark:border-transparent dark:bg-white/[0.06] dark:hover:bg-white/10"
        >
          <span
            className="min-w-0 truncate"
            style={{ fontFamily: `"${value ?? defaultFont}", var(--font-sans)` }}
          >
            {value ?? defaultLabel}
          </span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={2}
            className="size-4 shrink-0 text-muted-foreground"
          />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        className="w-64 gap-0 rounded-lg p-1 font-heading"
      >
        <Command className="rounded-none bg-transparent p-0">
          <CommandInput
            placeholder={t("settings.appearance.custom.fontSearch")}
          />
          <CommandList className="max-h-64 pt-1.5">
            <CommandEmpty>
              {t("settings.appearance.custom.fontNoResults")}
            </CommandEmpty>
            <CommandItem
              value={defaultLabel}
              onSelect={() => select(null)}
              data-checked={value === null}
              className="cursor-pointer rounded-[11px]"
            >
              <span style={{ fontFamily: `"${defaultFont}", var(--font-sans)` }}>
                {defaultLabel}
              </span>
            </CommandItem>
            <CommandGroup
              className="p-0"
              heading={t("settings.appearance.custom.fontBundledGroup")}
            >
              {BUNDLED_FONTS.map(renderItem)}
            </CommandGroup>
            {importedFonts.length > 0 && (
              <CommandGroup
                className="p-0"
                heading={t("settings.appearance.custom.fontImportedGroup")}
              >
                {importedFonts.map((font) => (
                  <CommandItem
                    key={font.name}
                    value={font.name}
                    onSelect={() => select(font.name)}
                    data-checked={value === font.name}
                    className="cursor-pointer rounded-[11px]"
                  >
                    <span
                      className="min-w-0 truncate"
                      style={{ fontFamily: `"${font.name}", var(--font-sans)` }}
                    >
                      {font.name}
                    </span>
                    <button
                      type="button"
                      aria-label={`${t("settings.appearance.custom.importFont.remove")}: ${font.name}`}
                      onPointerDown={(e) => e.stopPropagation()}
                      onClick={(e) => {
                        e.stopPropagation();
                        removeImportedFont(font.name);
                      }}
                      className="ml-auto rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground"
                    >
                      <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
                    </button>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
            {folderFonts.some((f) => !knownNames.has(f.name)) && (
              <CommandGroup
                className="p-0"
                heading={t("settings.appearance.custom.fontFolderGroup")}
              >
                {folderFonts
                  .filter((f) => !knownNames.has(f.name))
                  .map(({ name, file }) => (
                    <CommandItem
                      key={`folder-${name}`}
                      value={name}
                      onSelect={() => importFile(file)}
                      className="cursor-pointer rounded-[11px]"
                    >
                      <span className="min-w-0 truncate">{name}</span>
                    </CommandItem>
                  ))}
              </CommandGroup>
            )}
            <CommandGroup
              className="p-0"
              heading={t("settings.appearance.custom.fontDeviceGroup")}
            >
              {deviceFonts === null ? (
                <div className="px-3 py-2 text-xs text-muted-foreground">
                  {t("settings.appearance.custom.fontDeviceLoading")}
                </div>
              ) : (
                deviceOnlyFonts.map(renderItem)
              )}
            </CommandGroup>
          </CommandList>
        </Command>
        <div className="mt-1 flex items-center gap-1 border-t border-border/60 pt-1">
          <button
            type="button"
            onClick={requestUpload}
            className="flex flex-1 cursor-pointer items-center gap-2 whitespace-nowrap rounded-[11px] px-3 py-2 text-sm hover:bg-accent hover:text-accent-foreground"
          >
            <HugeiconsIcon
              icon={Upload01Icon}
              className="size-4 shrink-0 text-muted-foreground"
            />
            {t("settings.appearance.custom.importFont.upload")}
          </button>
          <div className="h-4 w-px shrink-0 bg-border" />
          <button
            type="button"
            onClick={requestFolder}
            className="flex flex-1 cursor-pointer items-center gap-2 whitespace-nowrap rounded-[11px] px-3 py-2 text-sm hover:bg-accent hover:text-accent-foreground"
          >
            <HugeiconsIcon
              icon={Folder01Icon}
              className="size-4 shrink-0 text-muted-foreground"
            />
            {t("settings.appearance.custom.importFont.scanFolder")}
          </button>
        </div>
        {uploadInputs}
      </PopoverContent>
    </Popover>
  );
}

export function UiFontRow() {
  const t = useT();
  const uiFont = useAppearanceCustomStore((s) => s.customization.uiFont);
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <FontSelect
      value={uiFont}
      defaultFont={DEFAULT_FONT_NAMES.ui}
      onCommit={(next) => patch({ uiFont: next })}
      ariaLabel={t("settings.appearance.custom.uiFont.label")}
    />
  );
}

export function HeadingFontRow() {
  const t = useT();
  const headingFont = useAppearanceCustomStore(
    (s) => s.customization.headingFont,
  );
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <FontSelect
      value={headingFont}
      defaultFont={DEFAULT_FONT_NAMES.heading}
      onCommit={(next) => patch({ headingFont: next })}
      ariaLabel={t("settings.appearance.custom.headingFont.label")}
    />
  );
}

export function ChatFontRow() {
  const t = useT();
  const chatFont = useAppearanceCustomStore((s) => s.customization.chatFont);
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <FontSelect
      value={chatFont}
      defaultFont={DEFAULT_FONT_NAMES.chat}
      onCommit={(next) => patch({ chatFont: next })}
      ariaLabel={t("settings.appearance.custom.chatFont.label")}
    />
  );
}

export function CodeFontRow() {
  const t = useT();
  const codeFont = useAppearanceCustomStore((s) => s.customization.codeFont);
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <FontSelect
      value={codeFont}
      defaultFont={DEFAULT_FONT_NAMES.code}
      onCommit={(next) => patch({ codeFont: next })}
      ariaLabel={t("settings.appearance.custom.codeFont.label")}
    />
  );
}

/* ---------------------------- Imported fonts ---------------------------- */

const FONT_MIME_BY_EXTENSION: Record<string, string> = {
  woff2: "font/woff2",
  woff: "font/woff",
  ttf: "font/ttf",
  otf: "font/otf",
};

const MAX_FONT_FILE_BYTES = Math.floor(1.5 * 1024 * 1024);

// Trailing style words stripped when matching a file name to a family the
// user already has ("Roboto-Bold.ttf" should match an installed "Roboto").
const FONT_STYLE_SUFFIXES = new Set([
  "regular",
  "bold",
  "italic",
  "light",
  "medium",
  "thin",
  "black",
  "semibold",
  "extrabold",
  "extralight",
  "heavy",
  "book",
  "oblique",
  "normal",
  "variable",
  "vf",
]);

function fontBaseName(fileName: string): string {
  return fileName
    .replace(/\.[^.]+$/, "")
    .replace(/[;{}()<>"']/g, "")
    .replace(/[_-]+/g, " ")
    .trim();
}

function familyCandidates(base: string): string[] {
  const candidates = [base];
  let words = base.split(/\s+/);
  while (
    words.length > 1 &&
    FONT_STYLE_SUFFIXES.has(words[words.length - 1].toLowerCase())
  ) {
    words = words.slice(0, -1);
    candidates.push(words.join(" "));
  }
  return candidates;
}

/* ----------------------------- Folder fonts ----------------------------- */

// Font files found by a folder scan this session, shared by every dropdown.
// File handles cannot be persisted from a plain directory input, so the list
// lives for the session; picking one imports it through the normal path.
type FolderFont = { name: string; file: File };
let folderFontsCache: FolderFont[] = [];
const folderFontsListeners = new Set<() => void>();

function setFolderFonts(next: FolderFont[]) {
  folderFontsCache = next;
  for (const listener of folderFontsListeners) listener();
}

function useFolderFonts(): FolderFont[] {
  return useSyncExternalStore(
    (onChange) => {
      folderFontsListeners.add(onChange);
      return () => folderFontsListeners.delete(onChange);
    },
    () => folderFontsCache,
  );
}

function fontNameFromFile(fileName: string, taken: Set<string>): string {
  const base = fontBaseName(fileName).slice(0, 60) || "Imported font";
  let candidate = base;
  let suffix = 2;
  while (taken.has(candidate)) {
    candidate = `${base} ${suffix}`;
    suffix += 1;
  }
  return candidate;
}

/**
 * File-import plumbing for the font dropdowns: hidden inputs, validation,
 * and persistence. Successful uploads call onImported with the font name
 * so the dropdown can select it for its slot right away. Fonts the user
 * already has (bundled, imported, or installed on the device) are selected
 * directly instead of embedding a duplicate copy.
 */
function useFontImport(onImported: (name: string) => void) {
  const t = useT();
  const importedFonts = useAppearanceCustomStore(
    (s) => s.customization.importedFonts,
  );
  const addImportedFont = useAppearanceCustomStore((s) => s.addImportedFont);
  const inputRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);

  // Match the file name against fonts that already exist somewhere.
  const findExisting = (fileName: string): string | null => {
    for (const candidate of familyCandidates(fontBaseName(fileName))) {
      const lower = candidate.toLowerCase();
      const imported = importedFonts.find((f) => f.name.toLowerCase() === lower);
      if (imported) return imported.name;
      const bundled = BUNDLED_FONTS.find((f) => f.toLowerCase() === lower);
      if (bundled) return bundled;
      if (detectFontsByMeasurement([candidate]).length > 0) return candidate;
    }
    return null;
  };

  const importFile = (file: File) => {
    const existing = findExisting(file.name);
    if (existing) {
      toast.info(t("settings.appearance.custom.importFont.alreadyAvailable"));
      onImported(existing);
      return;
    }
    if (importedFonts.length >= MAX_IMPORTED_FONTS) {
      toast.error(t("settings.appearance.custom.importFont.errorLimit"));
      return;
    }
    const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
    const mime = FONT_MIME_BY_EXTENSION[extension];
    if (!mime) {
      toast.error(t("settings.appearance.custom.importFont.errorInvalidType"));
      return;
    }
    if (file.size > MAX_FONT_FILE_BYTES) {
      toast.error(t("settings.appearance.custom.importFont.errorTooLarge"));
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => {
      toast.error(t("settings.appearance.custom.importFont.errorFailed"));
    };
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        toast.error(t("settings.appearance.custom.importFont.errorFailed"));
        return;
      }
      // Rewrite whatever MIME the browser guessed to the extension's font
      // type so the stored data URL passes frontend and backend validation.
      const base64 = result.slice(result.indexOf(",") + 1);
      const dataUrl = `data:${mime};base64,${base64}`;
      // Keep the persisted store under the localStorage quota.
      const existingTotal = importedFonts.reduce(
        (sum, f) => sum + f.dataUrl.length,
        0,
      );
      if (
        existingTotal + dataUrl.length >
        MAX_TOTAL_IMPORTED_FONT_DATA_URL_LENGTH
      ) {
        toast.error(
          t("settings.appearance.custom.importFont.errorStorageFull"),
        );
        return;
      }
      const taken = new Set<string>([
        ...BUNDLED_FONTS,
        ...importedFonts.map((f) => f.name),
      ]);
      const name = fontNameFromFile(file.name, taken);
      // Prove the file is a loadable font before persisting it.
      const face = new FontFace(name, `url(${dataUrl})`);
      face
        .load()
        .then(() => {
          addImportedFont({ name, dataUrl });
          onImported(name);
        })
        .catch(() => {
          toast.error(t("settings.appearance.custom.importFont.errorFailed"));
        });
    };
    reader.readAsDataURL(file);
  };

  const scanFolder = (files: FileList) => {
    const found: FolderFont[] = [];
    const seen = new Set<string>();
    for (const file of Array.from(files)) {
      const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
      if (!FONT_MIME_BY_EXTENSION[extension]) continue;
      const name = fontBaseName(file.name).slice(0, 60);
      if (!name || seen.has(name.toLowerCase())) continue;
      seen.add(name.toLowerCase());
      found.push({ name, file });
      if (found.length >= 200) break;
    }
    if (found.length === 0) {
      toast.info(t("settings.appearance.custom.importFont.folderNoFonts"));
      return;
    }
    found.sort((a, b) => a.name.localeCompare(b.name));
    setFolderFonts(found);
  };

  const requestUpload = () => inputRef.current?.click();
  const requestFolder = () => folderRef.current?.click();

  const inputs = (
    <>
      <input
        ref={inputRef}
        type="file"
        accept=".woff2,.woff,.ttf,.otf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) importFile(file);
          e.target.value = "";
        }}
      />
      <input
        ref={folderRef}
        type="file"
        className="hidden"
        // Non-standard but universal directory picker attributes.
        {...({ webkitdirectory: "", directory: "" } as object)}
        onChange={(e) => {
          if (e.target.files) scanFolder(e.target.files);
          e.target.value = "";
        }}
      />
    </>
  );

  return { inputs, requestUpload, requestFolder, importFile };
}

function FontSizeInput({
  value,
  range,
  onCommit,
  ariaLabel,
}: {
  value: number | null;
  range: { min: number; max: number; default: number };
  onCommit: (next: number | null) => void;
  ariaLabel: string;
}) {
  const [draft, setDraft] = useState(value === null ? "" : String(value));
  useEffect(() => {
    setDraft(value === null ? "" : String(value));
  }, [value]);
  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed === "") {
      onCommit(null);
      return;
    }
    const parsed = Number.parseInt(trimmed, 10);
    if (Number.isNaN(parsed)) {
      setDraft(value === null ? "" : String(value));
      return;
    }
    onCommit(Math.min(range.max, Math.max(range.min, parsed)));
  };
  return (
    <div className="flex items-center gap-1.5">
      <Input
        type="number"
        inputMode="numeric"
        min={range.min}
        max={range.max}
        value={draft}
        placeholder={String(range.default)}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            commit();
            e.currentTarget.blur();
          }
        }}
        aria-label={ariaLabel}
        className="h-8 w-20 text-xs"
      />
      <span className="text-xs text-muted-foreground">px</span>
    </div>
  );
}

export function UiFontSizeRow() {
  const t = useT();
  const uiFontSize = useAppearanceCustomStore(
    (s) => s.customization.uiFontSize,
  );
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <FontSizeInput
      value={uiFontSize}
      range={UI_FONT_SIZE_RANGE}
      onCommit={(next) => patch({ uiFontSize: next })}
      ariaLabel={t("settings.appearance.custom.uiFontSize.label")}
    />
  );
}

export function CodeFontSizeRow() {
  const t = useT();
  const codeFontSize = useAppearanceCustomStore(
    (s) => s.customization.codeFontSize,
  );
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <FontSizeInput
      value={codeFontSize}
      range={CODE_FONT_SIZE_RANGE}
      onCommit={(next) => patch({ codeFontSize: next })}
      ariaLabel={t("settings.appearance.custom.codeFontSize.label")}
    />
  );
}

export function FontSmoothingSwitch() {
  const fontSmoothing = useAppearanceCustomStore(
    (s) => s.customization.fontSmoothing,
  );
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <Switch
      checked={fontSmoothing}
      onCheckedChange={(checked) => patch({ fontSmoothing: checked })}
    />
  );
}

/* ------------------------------ Interface ------------------------------- */

export function ContrastSliderRow() {
  const t = useT();
  const contrast = useAppearanceCustomStore((s) => s.customization.contrast);
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <div className="flex w-48 items-center gap-3">
      <Slider
        value={[contrast]}
        min={0}
        max={100}
        step={5}
        onValueChange={(values: number[]) => patch({ contrast: values[0] })}
        aria-label={t("settings.appearance.custom.contrast.label")}
      />
      <span className="w-8 text-right text-xs tabular-nums text-muted-foreground">
        {contrast}
      </span>
    </div>
  );
}

export function PointerCursorsSwitch() {
  const pointerCursors = useAppearanceCustomStore(
    (s) => s.customization.pointerCursors,
  );
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <Switch
      checked={pointerCursors}
      onCheckedChange={(checked) => patch({ pointerCursors: checked })}
    />
  );
}

const REDUCE_MOTION_OPTIONS: {
  value: ReduceMotionSetting;
  labelKey: TranslationKey;
}[] = [
  {
    value: "system",
    labelKey: "settings.appearance.custom.reduceMotion.system",
  },
  { value: "on", labelKey: "settings.appearance.custom.reduceMotion.on" },
  { value: "off", labelKey: "settings.appearance.custom.reduceMotion.off" },
];

export function ReduceMotionSegmented() {
  const t = useT();
  const reduceMotion = useAppearanceCustomStore(
    (s) => s.customization.reduceMotion,
  );
  const patch = useAppearanceCustomStore((s) => s.patch);
  const reduced = useReducedMotionConfig();
  return (
    <div className="hub-tab-toggle inline-flex h-8 items-center rounded-full">
      {REDUCE_MOTION_OPTIONS.map((opt) => {
        const active = reduceMotion === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => patch({ reduceMotion: opt.value })}
            aria-pressed={active}
            className={cn(
              "relative flex h-8 items-center rounded-full px-3 text-xs font-medium transition-colors",
              active
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId="reduce-motion-pill"
                className="hub-tab-toggle-pill absolute inset-0 rounded-full"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 500, damping: 35, mass: 0.5 }
                }
              />
            )}
            <span className="relative z-10">{t(opt.labelKey)}</span>
          </button>
        );
      })}
    </div>
  );
}

/* ------------------------------- Reset all ------------------------------ */

export function ResetCustomizationButton() {
  const t = useT();
  const customization = useAppearanceCustomStore((s) => s.customization);
  const resetAll = useAppearanceCustomStore((s) => s.resetAll);
  const pristine = isDefaultCustomization(customization);
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      disabled={pristine}
      onClick={resetAll}
    >
      {t("settings.appearance.custom.resetAll")}
    </Button>
  );
}

export { DEFAULT_CUSTOMIZATION };
