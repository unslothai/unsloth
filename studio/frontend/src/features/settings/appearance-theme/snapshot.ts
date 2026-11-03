// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  beginBackgroundUpload,
  finishBackgroundUpload,
  hydrateBackground,
  resetChatBackground,
  sanitizeAdjustments,
  uploadBackgroundImage,
  useBackgroundStore,
  useMaterialStore,
} from "@/features/chat-background";
import { decodeBase64Bytes } from "@/lib/base64-bytes";
import { useAppearanceCustomStore } from "../stores/appearance-custom-store.ts";
import { getPalette, setPalette } from "../stores/theme-store.ts";
import type { AppearanceSnapshot } from "./application/plan-apply.ts";
import type { ThemeBackground } from "./domain/schema.ts";

export type AppearanceTarget = Omit<AppearanceSnapshot, "background">;

/** Like readAppearanceSnapshot, after the stored background has loaded from IndexedDB. */
export async function readLoadedAppearanceSnapshot(): Promise<AppearanceSnapshot> {
  await hydrateBackground();
  return readAppearanceSnapshot();
}

/** True while `now` still shows the exact background (image and adjustments) in `applied`. */
export function sameBackground(
  now: AppearanceSnapshot["background"],
  applied: AppearanceSnapshot["background"],
): boolean {
  return Boolean(now && applied && now.blob === applied.blob &&
    JSON.stringify(now.adjustments) === JSON.stringify(applied.adjustments));
}

/** Reads the live appearance. Does not write anything. */
export function readAppearanceSnapshot(): AppearanceSnapshot {
  const { material, style, strength, lens, menus } = useMaterialStore.getState();
  const background = useBackgroundStore.getState().value;
  return {
    palette: getPalette(),
    customization: useAppearanceCustomStore.getState().customization,
    glass: { material, style, strength, lens, menus },
    background: background
      ? { adjustments: sanitizeAdjustments(background), blob: background.blob }
      : null,
  };
}

/**
 * Writes palette, customization and glass in one pass; the background is separate. Theme mode and
 * interface scale are never touched. A native-only material off macOS is kept and shown as solid,
 * as the glass controls already do.
 */
export function applyAppearanceSnapshot(next: AppearanceTarget): void {
  useAppearanceCustomStore.getState().replaceAll(next.customization);
  setPalette(next.palette);
  const { material, style, strength, lens, menus } = next.glass;
  useMaterialStore.setState({ material, style, strength, lens, menus, nativeFailed: false });
}

/** Puts back a background read by readAppearanceSnapshot; null means none. Resolves false if it could not be stored. */
export function restoreBackground(prev: AppearanceSnapshot["background"]): Promise<boolean> {
  if (!prev) return resetChatBackground();
  return finishBackgroundUpload(beginBackgroundUpload(), { ...prev.adjustments, blob: prev.blob });
}

/** Stores a theme's background through the normal upload checks. Resolves false when it can't. */
export async function applyThemeBackground(background: ThemeBackground): Promise<boolean> {
  // The parser only lets through data:image/(png|jpeg|webp);base64,<payload>.
  const { dataUrl } = background;
  let blob: Blob;
  try {
    blob = new Blob([decodeBase64Bytes(dataUrl.slice(dataUrl.indexOf(",") + 1))], {
      type: dataUrl.slice(5, dataUrl.indexOf(";")),
    });
  } catch {
    return false;
  }
  return uploadBackgroundImage(blob, sanitizeAdjustments(background.adjustments));
}
