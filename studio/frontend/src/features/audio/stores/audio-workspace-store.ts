// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { AudioWorkflowId } from "../workflows";

/** Only the page commits `workflow`; the sidebar and deep links put a request in
 *  `requestedWorkflow`, which the page takes once it is free to switch, like an in-page tab click. */
interface AudioWorkspaceState {
  workflow: AudioWorkflowId;
  workflowChosen: boolean;
  requestedWorkflow: AudioWorkflowId | null;
  navExpanded: boolean;
  commitWorkflow: (workflow: AudioWorkflowId) => void;
  adoptWorkflow: (workflow: AudioWorkflowId) => void;
  requestWorkflow: (workflow: AudioWorkflowId) => void;
  clearRequestedWorkflow: () => void;
  setNavExpanded: (expanded: boolean) => void;
}

export const useAudioWorkspaceStore = create<AudioWorkspaceState>()((set) => ({
  workflow: "speak",
  workflowChosen: false,
  requestedWorkflow: null,
  navExpanded: false,
  commitWorkflow: (workflow) => set({ workflow, workflowChosen: true }),
  adoptWorkflow: (workflow) =>
    set((state) => (state.workflowChosen ? state : { workflow })),
  requestWorkflow: (requestedWorkflow) => set({ requestedWorkflow }),
  clearRequestedWorkflow: () => set({ requestedWorkflow: null }),
  setNavExpanded: (navExpanded) => set({ navExpanded }),
}));
