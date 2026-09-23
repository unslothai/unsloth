// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/**
 * One fork at a time, across every caller.
 *
 * The chord, the message button and the row menu each hold their own instance, so a `useState`
 * flag only disables the one that was used: pressing the chord and then clicking Fork before the
 * first request lands would post two, each with its own new thread id, and race their
 * navigations. A store is what all of them read.
 */
export const useForkInFlight = create<{
  forking: boolean;
  setForking: (forking: boolean) => void;
}>((set) => ({
  forking: false,
  setForking: (forking) => set({ forking }),
}));
