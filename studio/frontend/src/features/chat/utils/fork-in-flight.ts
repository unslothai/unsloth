// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/** One fork at a time across chord, message button and row menu; a store all of them read. */
export const useForkInFlight = create<{
  forking: boolean;
  setForking: (forking: boolean) => void;
}>((set) => ({
  forking: false,
  setForking: (forking) => set({ forking }),
}));
