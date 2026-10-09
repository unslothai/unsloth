// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioWorkflowId } from "../workflows";

export const AUDIO_WORKSPACE_STORAGE_KEY = "unsloth_audio_workspace";

/** Only the page commits `workflow`; the sidebar and deep links put a request in
 *  `requestedWorkflow`, which the page takes once it is free to switch, like an in-page tab click. */
interface AudioWorkspaceState {
  workflow: AudioWorkflowId;
  workflowChosen: boolean;
  requestedWorkflow: AudioWorkflowId | null;
  navExpanded: boolean;
  lastModelByWorkflow: Partial<Record<AudioWorkflowId, string>>;
  commitWorkflow: (workflow: AudioWorkflowId) => void;
  adoptWorkflow: (workflow: AudioWorkflowId) => void;
  requestWorkflow: (workflow: AudioWorkflowId) => void;
  clearRequestedWorkflow: () => void;
  setNavExpanded: (expanded: boolean) => void;
  rememberModel: (workflow: AudioWorkflowId, id: string) => void;
}

export const useAudioWorkspaceStore = create<AudioWorkspaceState>()(
  persist(
    (set) => ({
      workflow: "speak",
      workflowChosen: false,
      requestedWorkflow: null,
      navExpanded: false,
      lastModelByWorkflow: {},
      // Any committed switch settles a pending request, so a stale one never fires later.
      commitWorkflow: (workflow) =>
        set({ workflow, workflowChosen: true, requestedWorkflow: null }),
      adoptWorkflow: (workflow) =>
        set((state) => (state.workflowChosen ? state : { workflow })),
      requestWorkflow: (requestedWorkflow) => set({ requestedWorkflow }),
      clearRequestedWorkflow: () => set({ requestedWorkflow: null }),
      setNavExpanded: (navExpanded) => set({ navExpanded }),
      rememberModel: (workflow, id) =>
        set((state) =>
          state.lastModelByWorkflow[workflow] === id
            ? state
            : {
                lastModelByWorkflow: {
                  ...state.lastModelByWorkflow,
                  [workflow]: id,
                },
              },
        ),
    }),
    {
      name: AUDIO_WORKSPACE_STORAGE_KEY,
      version: 1,
      partialize: (state) => ({
        lastModelByWorkflow: state.lastModelByWorkflow,
      }),
    },
  ),
);
