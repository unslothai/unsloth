// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { WorkflowId } from "../workflows";

/** A trained adapter for Images to load, as its Train page's Deploy to Create does. */
export interface AdapterDeployRequest {
  baseRepo: string;
  family: string;
  catalogPath: string;
  trigger: string;
}

/** `supported` null means nothing is loaded, so every workflow stays selectable. */
interface ImageWorkflowState {
  workflow: WorkflowId;
  pageMode: "create" | "train";
  supported: WorkflowId[] | null;
  navExpanded: boolean;
  setNavExpanded: (expanded: boolean) => void;
  setWorkflow: (id: WorkflowId) => void;
  setPageMode: (mode: "create" | "train") => void;
  setSupported: (ids: WorkflowId[] | null) => void;
  /** From Video's Train tab, which shares the trainer but loads no adapters. */
  pendingDeploy: AdapterDeployRequest | null;
  requestDeploy: (request: AdapterDeployRequest) => void;
  takeDeploy: () => AdapterDeployRequest | null;
  /** Bumped by Video's Train tab so Images refetches its adapter list. */
  trainingCompletions: number;
  noteTrainingComplete: () => void;
}

export const useImageWorkflowStore = create<ImageWorkflowState>((set, get) => ({
  workflow: "create",
  pageMode: "create",
  supported: null,
  navExpanded: false,
  setNavExpanded: (navExpanded) => set({ navExpanded }),
  setWorkflow: (workflow) => set({ workflow, pageMode: "create" }),
  setPageMode: (pageMode) => set({ pageMode }),
  setSupported: (supported) => set({ supported }),
  pendingDeploy: null,
  requestDeploy: (pendingDeploy) => set({ pendingDeploy }),
  takeDeploy: () => {
    const request = get().pendingDeploy;
    if (request) set({ pendingDeploy: null });
    return request;
  },
  trainingCompletions: 0,
  noteTrainingComplete: () =>
    set((s) => ({ trainingCompletions: s.trainingCompletions + 1 })),
}));

export function isWorkflowEnabled(
  id: WorkflowId,
  supported: WorkflowId[] | null,
): boolean {
  return supported === null || supported.includes(id);
}
