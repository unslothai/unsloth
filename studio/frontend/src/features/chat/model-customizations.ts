// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { syncAcrossWindows } from "../../lib/sync-across-windows.ts";
import { hfCacheRepoId } from "../hub/lib/model-identity.ts";

// Labels only: loads and requests always use the model id.
export const MODEL_CUSTOMIZATIONS_KEY = "unsloth_model_customizations";
export const MODEL_CUSTOM_NAME_MAX = 80;
export const MODEL_ICON_PX = 64;
// A 64px PNG data URL is far below this; anything larger was not written here.
const MODEL_ICON_MAX_CHARS = 64_000;

export interface ModelCustomization {
  name?: string;
  icon?: string;
}

// A model adopted from the backend can carry its HF cache snapshot path instead of the repo id.
export function modelCustomizationKey(id: string | null | undefined): string {
  const trimmed = id?.trim() ?? "";
  return (hfCacheRepoId(trimmed) ?? trimmed).toLowerCase();
}

export function normaliseModelCustomization(
  value: unknown,
): ModelCustomization | null {
  if (!value || typeof value !== "object") return null;
  const { name, icon } = value as Record<string, unknown>;
  const out: ModelCustomization = {};
  if (typeof name === "string" && name.trim()) {
    out.name = name.trim().slice(0, MODEL_CUSTOM_NAME_MAX);
  }
  if (
    typeof icon === "string" &&
    icon.startsWith("data:image/") &&
    icon.length <= MODEL_ICON_MAX_CHARS
  ) {
    out.icon = icon;
  }
  return out.name || out.icon ? out : null;
}

function normaliseAll(value: unknown): Record<string, ModelCustomization> {
  const out: Record<string, ModelCustomization> = {};
  if (!value || typeof value !== "object") return out;
  for (const [id, entry] of Object.entries(value)) {
    const key = modelCustomizationKey(id);
    const clean = normaliseModelCustomization(entry);
    if (key && clean) out[key] = clean;
  }
  return out;
}

interface ModelCustomizationsState {
  byId: Record<string, ModelCustomization>;
  /** Empty name and icon clear the entry. */
  setCustomization: (id: string, value: ModelCustomization) => void;
}

const safeStorage = createJSONStorage(() => ({
  getItem: (name: string) => localStorage.getItem(name),
  setItem: (name: string, value: string) => {
    try {
      localStorage.setItem(name, value);
    } catch {
      // Kept in memory for this session.
    }
  },
  removeItem: (name: string) => {
    try {
      localStorage.removeItem(name);
    } catch {
      // Same as a failed write.
    }
  },
}));

export const useModelCustomizationsStore = create<ModelCustomizationsState>()(
  persist(
    (set) => ({
      byId: {},
      setCustomization: (id, value) =>
        set((state) => {
          const key = modelCustomizationKey(id);
          if (!key) return state;
          const next = normaliseModelCustomization(value);
          const prev = state.byId[key];
          if (next?.name === prev?.name && next?.icon === prev?.icon) {
            return state;
          }
          const byId = { ...state.byId };
          if (next) byId[key] = next;
          else delete byId[key];
          return { byId };
        }),
    }),
    {
      name: MODEL_CUSTOMIZATIONS_KEY,
      storage: safeStorage,
      partialize: (state) => ({ byId: state.byId }),
      merge: (persisted, current) => ({
        ...current,
        byId: normaliseAll((persisted as { byId?: unknown } | undefined)?.byId),
      }),
    },
  ),
);

syncAcrossWindows(useModelCustomizationsStore);

export function useModelCustomization(
  id: string | null | undefined,
): ModelCustomization | undefined {
  const key = modelCustomizationKey(id);
  return useModelCustomizationsStore((state) =>
    key ? state.byId[key] : undefined,
  );
}

/** Square-crops and downsizes so a photo fits in local storage. */
export async function imageFileToIconDataUrl(file: Blob): Promise<string> {
  if (!file.type.startsWith("image/")) {
    throw new Error("Choose an image file.");
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    const side = Math.min(width, height);
    if (!side) throw new Error("This image has no size.");
    const canvas = document.createElement("canvas");
    canvas.width = MODEL_ICON_PX;
    canvas.height = MODEL_ICON_PX;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not read this image.");
    context.drawImage(
      image,
      (width - side) / 2,
      (height - side) / 2,
      side,
      side,
      0,
      0,
      MODEL_ICON_PX,
      MODEL_ICON_PX,
    );
    const icon = canvas.toDataURL("image/png");
    if (!normaliseModelCustomization({ icon })) {
      throw new Error("Could not read this image.");
    }
    return icon;
  } finally {
    URL.revokeObjectURL(url);
  }
}
