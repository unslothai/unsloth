// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so node:test can load it.

/** Ids are stored per account: a new page needs a new id, and only accounts that never saw it get it. */
export const ONBOARDING_PAGES = [
  "welcome",
  "profile",
  "look",
  "scene",
  "device",
  "start",
  "done",
] as const;

export type OnboardingPageId = (typeof ONBOARDING_PAGES)[number];

/** Bookends carry no progress dot and are never shown on their own. */
const BOOKENDS: ReadonlySet<OnboardingPageId> = new Set(["welcome", "done"]);

export type OnboardingRecord = {
  saved: boolean;
  pages: string[];
  completedAt: number | null;
  hasChatHistory: boolean;
};

export type OnboardingDecision =
  | { kind: "show"; pages: OnboardingPageId[] }
  // An account that chatted before onboarding existed: record it as done without showing it.
  | { kind: "adopt" }
  | { kind: "none" };

export function decideOnboarding(record: OnboardingRecord): OnboardingDecision {
  if (!record.saved) {
    return record.hasChatHistory
      ? { kind: "adopt" }
      : { kind: "show", pages: [...ONBOARDING_PAGES] };
  }
  const seen = new Set(record.pages);
  const unseen = ONBOARDING_PAGES.filter(
    (page) => !BOOKENDS.has(page) && !seen.has(page),
  );
  if (unseen.length === 0) {
    // A run closed on the last page before Finish landed: nothing left to say.
    return { kind: "none" };
  }
  const pages: OnboardingPageId[] = seen.has("welcome") ? [] : ["welcome"];
  return { kind: "show", pages: [...pages, ...unseen, "done"] };
}

export function isNumberedPage(page: OnboardingPageId): boolean {
  return !BOOKENDS.has(page);
}

export const START_CHOICES = ["chat", "train", "recipes", "hub"] as const;
export type StartChoice = (typeof START_CHOICES)[number];

export const START_ROUTES: Record<
  StartChoice,
  "/chat" | "/studio" | "/data-recipes" | "/hub"
> = {
  chat: "/chat",
  train: "/studio",
  recipes: "/data-recipes",
  hub: "/hub",
};

/** Guided tour ids from features/tour/lib/tour-routes.ts. */
export const START_TOURS: Record<StartChoice, string> = {
  chat: "chat",
  train: "studio",
  recipes: "data-recipes",
  hub: "hub",
};

/** The greeting addresses the first word, as useEffectiveProfile does. */
export function greetingName(displayName: string): string {
  return displayName.trim().split(/\s+/)[0] ?? "";
}
