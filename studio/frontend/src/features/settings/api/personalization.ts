// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { AppearanceCustomization } from "../stores/appearance-custom-store";

export type PersonalizationProfile = {
  displayName: string;
  nickname: string;
  avatarDataUrl: string | null;
  avatarShape: "circle" | "rounded";
  showGreetingSloth: boolean;
};

export type PersonalizationAppearance = {
  theme: "light" | "dark" | "system";
  palette: "standard" | "classic" | "minimal";
  language: string | null;
  customization: AppearanceCustomization;
};

export type Personalization = {
  version: number;
  profile: PersonalizationProfile;
  appearance: PersonalizationAppearance;
  // Distinguishes server hydrate from first local migration.
  saved: boolean;
  // False when the stored record predates these fields (legacy migration): the
  // client then keeps local values instead of the server-filled defaults.
  customizationSaved: boolean;
  paletteSaved: boolean;
  greetingSlothSaved: boolean;
};

export async function loadPersonalization(): Promise<Personalization> {
  const res = await authFetch("/api/settings/personalization");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load personalization"),
    );
  }
  return (await res.json()) as Personalization;
}

export async function savePersonalization(
  data: Omit<
    Personalization,
    "saved" | "customizationSaved" | "paletteSaved" | "greetingSlothSaved"
  >,
): Promise<void> {
  const res = await authFetch("/api/settings/personalization", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save personalization"),
    );
  }
}
