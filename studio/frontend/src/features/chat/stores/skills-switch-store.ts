// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// Skills switch: off hides the pill and skill tools; unset follows whether a skill is ticked.
interface SkillsSwitchState {
  on: boolean | null;
  setOn: (on: boolean | null) => void;
}

export const useSkillsSwitchStore = create<SkillsSwitchState>()(
  persist((set) => ({ on: null, setOn: (on) => set({ on }) }), {
    name: "unsloth_chat_skills_on",
  }),
);

export function skillsSwitchedOn(
  on: boolean | null,
  enabledCount: number,
): boolean {
  return on ?? enabledCount > 0;
}
