// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { ONBOARDING_PAGES, type OnboardingPageId } from "./lib/pages";

type OnboardingState = {
  pages: OnboardingPageId[] | null;
  show: (pages: OnboardingPageId[]) => void;
  close: () => void;
};

export const useOnboardingStore = create<OnboardingState>()((set) => ({
  pages: null,
  show: (pages) => set({ pages }),
  close: () => set({ pages: null }),
}));

/** Settings replays every page, whatever the account has seen. */
export function openOnboarding(): void {
  useOnboardingStore.getState().show([...ONBOARDING_PAGES]);
}
