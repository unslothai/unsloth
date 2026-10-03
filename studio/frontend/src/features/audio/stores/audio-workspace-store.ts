// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioWorkflowId } from "../workflows";

export const AUDIO_WORKSPACE_STORAGE_KEY = "unsloth_audio_workspace";

/** The Audio page's workflow, lifted out of the page so the sidebar and the More flyout can drive it.
 *  Only the page commits `workflow`; the sidebar and deep links put a request in `requestedWorkflow`
 *  and the page takes it when it is free to switch, the same gate an in-page tab click passes. */
interface AudioWorkspaceState {
  workflow: AudioWorkflowId;
  /** Whether anything has picked the workflow yet this session, so a first look at the loaded
   *  model may still open the page that fits it. */
  workflowChosen: boolean;
  requestedWorkflow: AudioWorkflowId | null;
  /** Off the Audio page, whether the sidebar lists the workflows under the row. */
  navExpanded: boolean;
  /** The model last loaded on each workflow. */
  lastModelByWorkflow: Partial<Record<AudioWorkflowId, string>>;
  commitWorkflow: (workflow: AudioWorkflowId) => void;
  /** Opens a workflow only while nothing has chosen one. */
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
      commitWorkflow: (workflow) => set({ workflow, workflowChosen: true }),
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
      // The open workflow is not remembered across reloads, like the page's mode before it.
      partialize: (state) => ({
        lastModelByWorkflow: state.lastModelByWorkflow,
      }),
    },
  ),
);
