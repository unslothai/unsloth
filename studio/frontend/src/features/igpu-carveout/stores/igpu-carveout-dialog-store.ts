// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { dismissCarveoutNotice } from "../api/igpu-carveout-notice";
import { parseCarveoutAdvice, type IgpuCarveoutAdvice } from "../types";

interface IgpuCarveoutDialogStore {
  open: boolean;
  advice: IgpuCarveoutAdvice | null;
  /** Show the advice from a load response. No-op when there is none, which is
   *  almost every load, so callers can pass the field through unconditionally. */
  show: (value: unknown) => void;
  /** Close without recording anything: the notice may appear again on a later
   *  load. For the Escape key and the overlay, which are not decisions. */
  close: () => void;
  /** Close and stop offering it at this allocation. */
  dismissForever: () => void;
}

export const useIgpuCarveoutDialogStore = create<IgpuCarveoutDialogStore>((set, get) => ({
  open: false,
  advice: null,

  show: (value) => {
    const advice = parseCarveoutAdvice(value);
    // A load that arrives while the dialog is already up must not replace the
    // text under the user's cursor mid-read.
    if (get().open) return;
    if (!advice) {
      // Every load calls this, so a load carrying no advice is where the previous
      // load's numbers stop being true. Dropping them here rather than in `close`
      // is deliberate: `close` runs at the start of a 100ms exit animation, and
      // clearing then would blank the dialog's text instead of fading it out.
      if (get().advice) set({ advice: null });
      return;
    }
    set({ open: true, advice });
  },

  close: () => set({ open: false }),

  dismissForever: () => {
    const advice = get().advice;
    set({ open: false });
    // Fire and forget: the dialog is already gone, and dismissCarveoutNotice
    // swallows its own failures. Worst case the notice returns on a later load.
    void dismissCarveoutNotice(advice ? advice.current_gb : null);
  },
}));

/** Hand a load response's advice field to the dialog. Safe to call on every load. */
export function showCarveoutAdvice(value: unknown): void {
  useIgpuCarveoutDialogStore.getState().show(value);
}
