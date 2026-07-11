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
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import {
  CODE_FONT_SIZE_RANGE,
  type CustomModeColors,
  DEFAULT_CUSTOMIZATION,
  MAX_IMPORTED_FONTS,
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
      <span className="font-mono text-xs text-muted-foreground uppercase">
        {value}
      </span>
      <ColorPickerSwatch
        value={value}
        onChange={(hex) => setColor(resolved, colorKey, hex)}
        label={label}
        highlighted={override !== null}
      />
    </div>
  );
}

/* ------------------------------ Typography ------------------------------ */

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
  onCommit,
  ariaLabel,
}: {
  value: string | null;
  onCommit: (next: string | null) => void;
  ariaLabel: string;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [deviceFonts, setDeviceFonts] = useState<string[] | null>(
    deviceFontsCache,
  );
  const importedFonts = useAppearanceCustomStore(
    (s) => s.customization.importedFonts,
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
      <span className="min-w-0 truncate">{font}</span>
    </CommandItem>
  );

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          aria-label={ariaLabel}
          aria-expanded={open}
          className="flex h-8 w-56 cursor-pointer items-center justify-between gap-1.5 rounded-full border border-border bg-background px-3.5 text-xs outline-none transition-colors hover:bg-accent/50 focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 dark:border-white/10 dark:bg-white/[0.06] dark:hover:bg-white/10"
        >
          <span className="min-w-0 truncate">
            {value ?? t("settings.appearance.custom.fontDefault")}
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
        className="w-64 gap-0 rounded-xl p-1 font-heading corner-squircle"
      >
        <Command className="rounded-none bg-transparent p-0">
          <CommandInput
            placeholder={t("settings.appearance.custom.fontSearch")}
          />
          <CommandList className="max-h-64 pt-1">
            <CommandEmpty>
              {t("settings.appearance.custom.fontNoResults")}
            </CommandEmpty>
            <CommandItem
              value={t("settings.appearance.custom.fontDefault")}
              onSelect={() => select(null)}
              data-checked={value === null}
              className="cursor-pointer rounded-[11px]"
            >
              <span>{t("settings.appearance.custom.fontDefault")}</span>
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
                {importedFonts.map((font) => renderItem(font.name))}
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
      onCommit={(next) => patch({ uiFont: next })}
      ariaLabel={t("settings.appearance.custom.uiFont.label")}
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

function fontNameFromFile(fileName: string, taken: Set<string>): string {
  const base =
    fileName
      .replace(/\.[^.]+$/, "")
      .replace(/[;{}()<>"']/g, "")
      .replace(/[_-]+/g, " ")
      .trim()
      .slice(0, 60) || "Imported font";
  let candidate = base;
  let suffix = 2;
  while (taken.has(candidate)) {
    candidate = `${base} ${suffix}`;
    suffix += 1;
  }
  return candidate;
}

export function ImportFontControls() {
  const t = useT();
  const importedFonts = useAppearanceCustomStore(
    (s) => s.customization.importedFonts,
  );
  const addImportedFont = useAppearanceCustomStore((s) => s.addImportedFont);
  const removeImportedFont = useAppearanceCustomStore(
    (s) => s.removeImportedFont,
  );
  const inputRef = useRef<HTMLInputElement>(null);
  const atLimit = importedFonts.length >= MAX_IMPORTED_FONTS;

  const importFile = (file: File) => {
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
        })
        .catch(() => {
          toast.error(t("settings.appearance.custom.importFont.errorFailed"));
        });
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="flex flex-col items-end gap-2">
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
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => {
          if (atLimit) {
            toast.error(t("settings.appearance.custom.importFont.errorLimit"));
            return;
          }
          inputRef.current?.click();
        }}
      >
        {t("settings.appearance.custom.importFont.button")}
      </Button>
      {importedFonts.length > 0 && (
        <div className="flex max-w-72 flex-wrap justify-end gap-1.5">
          {importedFonts.map((font) => (
            <span
              key={font.name}
              className="inline-flex items-center gap-1 rounded-full border border-border px-2 py-1 text-xs text-foreground"
              style={{ fontFamily: `"${font.name}"` }}
            >
              {font.name}
              <button
                type="button"
                onClick={() => removeImportedFont(font.name)}
                aria-label={`${t("settings.appearance.custom.importFont.remove")}: ${font.name}`}
                className="text-muted-foreground transition-colors hover:text-foreground"
              >
                <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
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
        className="h-8 w-16 text-xs"
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

export function TranslucentSidebarSwitch() {
  const translucentSidebar = useAppearanceCustomStore(
    (s) => s.customization.translucentSidebar,
  );
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <Switch
      checked={translucentSidebar}
      onCheckedChange={(checked) => patch({ translucentSidebar: checked })}
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
  const reduced = useReducedMotion();
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
