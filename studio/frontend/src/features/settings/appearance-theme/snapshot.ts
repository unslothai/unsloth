// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  beginBackgroundUpload,
  dataUrlToBlob,
  failBackgroundUpload,
  finishBackgroundUpload,
  prepareBackgroundImage,
  resetChatBackground,
  sanitizeAdjustments,
  useBackgroundStore,
  useMaterialStore,
} from "@/features/chat-background";
import { useAppearanceCustomStore } from "../stores/appearance-custom-store.ts";
import { getPalette, setPalette } from "../stores/theme-store.ts";
import type { ThemeBackground } from "./domain/schema.ts";
import type { AppearanceSnapshot } from "./application/types.ts";

export type AppearanceTarget = Pick<
  AppearanceSnapshot,
  "palette" | "customization" | "glass"
>;

/** Reads the live appearance. Does not write anything. */
export function readAppearanceSnapshot(): AppearanceSnapshot {
  const material = useMaterialStore.getState();
  const background = useBackgroundStore.getState().value;
  return {
    palette: getPalette(),
    customization: useAppearanceCustomStore.getState().customization,
    glass: {
      material: material.material,
      strength: {
        light: material.strength.light,
        dark: material.strength.dark,
      },
      lens: material.lens,
      menus: material.menus,
    },
    background: background
      ? { adjustments: sanitizeAdjustments(background), blob: background.blob }
      : null,
  };
}

/**
 * Writes palette, customization and glass in one pass. The background is handled separately.
 * Theme mode and interface scale are never touched.
 *
 * A native-only material ("macos") is stored as given even off macOS. The window material mount
 * shows it as solid and makes no native call, and the glass controls treat it the same way.
 */
export function applyAppearanceSnapshot(next: AppearanceTarget): void {
  useAppearanceCustomStore.getState().replaceAll(next.customization);
  setPalette(next.palette);
  const material = useMaterialStore.getState();
  material.setMaterial(next.glass.material);
  material.setStrength(next.glass.strength.light, "light");
  material.setStrength(next.glass.strength.dark, "dark");
  material.setLens(next.glass.lens);
  material.setMenus(next.glass.menus);
}

/** Puts back a background read by readAppearanceSnapshot; null means no background. */
export async function restoreBackground(
  prev: AppearanceSnapshot["background"],
): Promise<void> {
  if (!prev) {
    await resetChatBackground();
    return;
  }
  const ticket = beginBackgroundUpload();
  await finishBackgroundUpload(ticket, {
    ...prev.adjustments,
    blob: prev.blob,
  });
}

/** Decodes and stores a theme's background. Returns false, and reports why, when it cannot. */
export async function applyThemeBackground(
  bg: ThemeBackground,
): Promise<boolean> {
  const ticket = beginBackgroundUpload();
  try {
    const blob = await prepareBackgroundImage(await dataUrlToBlob(bg.dataUrl));
    await finishBackgroundUpload(ticket, {
      ...sanitizeAdjustments(bg.adjustments),
      blob,
    });
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : "";
    failBackgroundUpload(
      ticket,
      message === "file" || message === "dimensions" ? message : "decode",
    );
    return false;
  }
}
