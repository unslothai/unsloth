// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  resetChatBackground,
  useBackgroundStore,
  useMaterialStore,
} from "@/features/chat-background";
import {
  setBackgroundCanvas,
  uploadBackgroundFile,
} from "@/features/chat-background/upload";
import {
  COLOR_THEMES,
  type ColorThemeId,
  FLAVOR_THEME_IDS,
  type ThemeModeColors,
} from "@/features/settings/lib/color-themes";
import {
  type Theme,
  usePalette,
  useTheme,
} from "@/features/settings/stores/theme-store";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  LaptopIcon,
  Moon02Icon,
  Sun03Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type ReactNode, useMemo, useRef, useState } from "react";
import {
  WALLPAPER_PRESETS,
  type WallpaperPreset,
  paintWallpaper,
} from "../lib/wallpapers";
import { ChosenRing, Group, Row, Segmented, StepHeader } from "./onboarding-ui";

export const MODES: {
  value: Theme;
  label: TranslationKey;
  icon: IconSvgElement;
}[] = [
  {
    value: "system",
    label: "settings.appearance.theme.system",
    icon: LaptopIcon,
  },
  { value: "light", label: "settings.appearance.theme.light", icon: Sun03Icon },
  { value: "dark", label: "settings.appearance.theme.dark", icon: Moon02Icon },
];

// One flavour per family, so the row reads as a range rather than near-duplicates.
const SWATCHES: ColorThemeId[] = [
  "standard",
  "matcha",
  "blueberry",
  "cherry",
  "tangerine",
  "taro",
  "honey",
  "neon-cyberpunk",
];

export function paletteName(id: ColorThemeId, standard: string): string {
  return COLOR_THEMES[id].name ?? standard;
}

function Miniature({ colors }: { colors: ThemeModeColors }) {
  const line = `color-mix(in oklab, ${colors.foreground} 22%, transparent)`;
  return (
    <div
      className="flex h-full w-full"
      style={{ backgroundColor: colors.background }}
    >
      <div
        className="flex h-full w-[30%] flex-col gap-1.5 p-2"
        style={{
          backgroundColor:
            colors.surface ??
            `color-mix(in oklab, ${colors.foreground} 5%, ${colors.background})`,
        }}
      >
        {[72, 52, 62].map((w, i) => (
          <div
            key={w}
            className="h-1 rounded-full"
            style={{
              width: `${w}%`,
              backgroundColor: i === 0 ? colors.foreground : line,
              opacity: i === 0 ? 0.35 : 1,
            }}
          />
        ))}
      </div>
      <div className="flex flex-1 flex-col gap-1.5 p-2.5">
        <div
          className="h-1 w-3/4 rounded-full"
          style={{ backgroundColor: line }}
        />
        <div
          className="h-1 w-1/2 rounded-full"
          style={{ backgroundColor: line }}
        />
        <div
          className="mt-auto flex h-3.5 items-center justify-end rounded-full border p-0.5"
          style={{ borderColor: line }}
        >
          <span
            className="h-full w-2.5 rounded-full"
            style={{ backgroundColor: colors.accent }}
          />
        </div>
      </div>
    </div>
  );
}

function ModeCard({
  mode,
  active,
  onPick,
  colors,
}: {
  mode: (typeof MODES)[number];
  active: boolean;
  onPick: () => void;
  colors: { light: ThemeModeColors; dark: ThemeModeColors };
}) {
  const t = useT();
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onPick}
      className="bg-card hover:bg-muted/60 relative flex flex-col rounded-2xl p-1 text-left ring-1 ring-border transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <span className="block aspect-[16/10] overflow-hidden rounded-[0.8rem]">
        {mode.value === "system" ? (
          <span className="flex h-full">
            <span className="h-full w-1/2 overflow-hidden">
              <span className="block h-full w-[200%]">
                <Miniature colors={colors.dark} />
              </span>
            </span>
            <span className="h-full w-1/2 overflow-hidden">
              <span className="-ml-[100%] block h-full w-[200%]">
                <Miniature colors={colors.light} />
              </span>
            </span>
          </span>
        ) : (
          <Miniature colors={colors[mode.value]} />
        )}
      </span>
      <span
        className={cn(
          "flex items-center justify-center gap-1.5 py-2 text-xs font-medium",
          active ? "text-foreground" : "text-muted-foreground",
        )}
      >
        <HugeiconsIcon icon={mode.icon} className="size-3.5" />
        {t(mode.label)}
      </span>
      <ChosenRing on={active} />
    </button>
  );
}

export function LookStep() {
  const t = useT();
  const { theme, resolved, setTheme } = useTheme();
  const { palette, setPalette } = usePalette();
  const standard = t("settings.appearance.palette.standard");
  const more = FLAVOR_THEME_IDS.length + 1 - SWATCHES.length;
  return (
    <div>
      <StepHeader
        title={t("onboarding.look.title")}
        body={t("onboarding.look.body")}
      />
      <div className="grid grid-cols-3 gap-3">
        {MODES.map((mode) => (
          <ModeCard
            key={mode.value}
            mode={mode}
            active={theme === mode.value}
            onPick={() => setTheme(mode.value)}
            colors={COLOR_THEMES[palette]}
          />
        ))}
      </div>
      <Group className="mt-6">
        <Row
          label={t("onboarding.look.theme")}
          description={paletteName(palette, standard)}
          trailing={
            <span className="text-muted-foreground text-xs">
              {t("onboarding.look.moreThemes", { count: more })}
            </span>
          }
          below={
            <div className="flex flex-wrap items-center gap-2">
              {SWATCHES.map((id) => {
                const colors = COLOR_THEMES[id][resolved];
                const active = palette === id;
                return (
                  <button
                    key={id}
                    type="button"
                    aria-pressed={active}
                    aria-label={paletteName(id, standard)}
                    title={paletteName(id, standard)}
                    onClick={() => setPalette(id)}
                    className="relative flex size-8 items-center justify-center rounded-full outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <span
                      className="flex size-6 items-center justify-center rounded-full border border-black/10 dark:border-white/15"
                      style={{ backgroundColor: colors.background }}
                    >
                      <span
                        className="size-3 rounded-full"
                        style={{ backgroundColor: colors.accent }}
                      />
                    </span>
                    <ChosenRing
                      on={active}
                      className="ring-foreground ring-[1.5px]"
                    />
                  </button>
                );
              })}
            </div>
          }
        />
      </Group>
    </div>
  );
}

const THUMB_W = 160;
const THUMB_H = 100;
const WALLPAPER_W = 2400;
const WALLPAPER_H = 1500;

function paintTo(
  width: number,
  height: number,
  preset: WallpaperPreset,
): HTMLCanvasElement | null {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  paintWallpaper(ctx, width, height, preset);
  return canvas;
}

function useThumbnails(): Record<string, string> {
  return useMemo(() => {
    const out: Record<string, string> = {};
    for (const preset of WALLPAPER_PRESETS) {
      const canvas = paintTo(THUMB_W, THUMB_H, preset);
      if (canvas) out[preset.id] = canvas.toDataURL("image/png");
    }
    return out;
  }, []);
}

type Backdrop = "none" | "upload" | string;

function WallpaperTile({
  active,
  label,
  onPick,
  children,
  busy,
}: {
  active: boolean;
  label: string;
  onPick: () => void;
  children: ReactNode;
  busy?: boolean;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      aria-label={label}
      title={label}
      onClick={onPick}
      className="group relative flex aspect-[4/3] items-center justify-center overflow-hidden rounded-xl ring-1 ring-border outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      {children}
      {busy ? (
        <span className="bg-background/60 absolute inset-0 flex items-center justify-center">
          <span className="border-muted-foreground/30 border-t-foreground size-4 animate-spin rounded-full border-2" />
        </span>
      ) : null}
      <ChosenRing on={active} className="ring-foreground ring-inset" />
    </button>
  );
}

export function SceneStep() {
  const t = useT();
  const thumbs = useThumbnails();
  const hasImage = useBackgroundStore((s) => s.value !== null);
  const busy = useBackgroundStore((s) => s.busy);
  const error = useBackgroundStore((s) => s.error);
  const material = useMaterialStore((s) => s.material);
  const setEnabled = useMaterialStore((s) => s.setEnabled);
  const [picked, setPicked] = useState<Backdrop>(hasImage ? "upload" : "none");
  const input = useRef<HTMLInputElement>(null);
  const current: Backdrop = hasImage ? picked : busy ? picked : "none";

  const pickPreset = (preset: WallpaperPreset) => {
    setPicked(preset.id);
    const canvas = paintTo(WALLPAPER_W, WALLPAPER_H, preset);
    if (canvas) void setBackgroundCanvas(canvas);
  };

  return (
    <div>
      <StepHeader
        title={t("onboarding.scene.title")}
        body={t("onboarding.scene.body")}
      />
      <div className="grid grid-cols-4 gap-2.5">
        <WallpaperTile
          active={current === "none"}
          label={t("onboarding.scene.none")}
          onPick={() => {
            setPicked("none");
            void resetChatBackground();
          }}
        >
          <span className="bg-background absolute inset-0" />
          <span className="text-muted-foreground relative text-xs font-medium">
            {t("onboarding.scene.none")}
          </span>
        </WallpaperTile>
        {WALLPAPER_PRESETS.map((preset) => (
          <WallpaperTile
            key={preset.id}
            active={current === preset.id}
            label={preset.name}
            busy={busy && picked === preset.id}
            onPick={() => pickPreset(preset)}
          >
            {thumbs[preset.id] ? (
              <img
                alt=""
                src={thumbs[preset.id]}
                className="absolute inset-0 size-full object-cover"
              />
            ) : (
              <span
                className="absolute inset-0"
                style={{ backgroundColor: preset.base }}
              />
            )}
          </WallpaperTile>
        ))}
        <WallpaperTile
          active={current === "upload"}
          label={t("onboarding.scene.upload")}
          busy={busy && picked === "upload"}
          onPick={() => input.current?.click()}
        >
          <span className="bg-muted/60 absolute inset-0" />
          <span className="text-muted-foreground relative flex flex-col items-center gap-1 text-xs font-medium">
            <HugeiconsIcon
              icon={Upload01Icon}
              className="size-4"
              strokeWidth={1.75}
            />
            {t("onboarding.scene.upload")}
          </span>
        </WallpaperTile>
        <input
          ref={input}
          className="sr-only"
          tabIndex={-1}
          aria-hidden="true"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (!file) return;
            setPicked("upload");
            void uploadBackgroundFile(file);
          }}
        />
      </div>
      {error ? (
        <p role="alert" className="text-destructive mt-3 text-center text-xs">
          {t("onboarding.scene.error")}
        </p>
      ) : null}
      <Group className="mt-6">
        <Row
          label={t("onboarding.scene.glass")}
          description={t(
            material === "solid"
              ? "onboarding.scene.solidHint"
              : "onboarding.scene.glassHint",
          )}
          trailing={
            <Segmented
              id="onboarding-glass"
              label={t("onboarding.scene.glass")}
              value={material === "solid" ? "solid" : "glass"}
              onChange={(next) => setEnabled(next === "glass")}
              options={[
                { value: "solid", label: t("onboarding.scene.solid") },
                { value: "glass", label: t("onboarding.scene.glassOn") },
              ]}
            />
          }
        />
      </Group>
    </div>
  );
}
