// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so node:test can load it.

/** Ids are stored per account: a new page needs a new id, and only accounts that never saw it get it. */
export const ONBOARDING_PAGES = ["welcome", "look", "done"] as const;

export type OnboardingPageId = (typeof ONBOARDING_PAGES)[number];

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

export const CHAT_TOUR = "chat";

/** The greeting addresses the first word, as useEffectiveProfile does. */
export function greetingName(displayName: string): string {
  return displayName.trim().split(/\s+/)[0] ?? "";
}

/** Video is gated on its own verdict, which a CPU host can pass (experimental CPU video) while chat-only. */
export function mediaAvailable(
  chatOnly: boolean,
  videoSupported: boolean | null,
): boolean {
  return !chatOnly || videoSupported === true;
}
