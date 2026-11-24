// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { OnboardingPageId, OnboardingRecord } from "./lib/pages";

export async function loadOnboarding(): Promise<OnboardingRecord> {
  const res = await authFetch("/api/settings/onboarding");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to load onboarding"));
  }
  return (await res.json()) as OnboardingRecord;
}

export async function saveOnboarding(
  pages: OnboardingPageId[],
  completed = false,
): Promise<void> {
  const res = await authFetch("/api/settings/onboarding", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pages, completed }),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to save onboarding"));
  }
}
