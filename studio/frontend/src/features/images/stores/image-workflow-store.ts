// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { WorkflowId } from "../workflows";

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
}

export const useImageWorkflowStore = create<ImageWorkflowState>((set) => ({
  workflow: "create",
  pageMode: "create",
  supported: null,
  navExpanded: false,
  setNavExpanded: (navExpanded) => set({ navExpanded }),
  setWorkflow: (workflow) => set({ workflow, pageMode: "create" }),
  setPageMode: (pageMode) => set({ pageMode }),
  setSupported: (supported) => set({ supported }),
}));

export function isWorkflowEnabled(
  id: WorkflowId,
  supported: WorkflowId[] | null,
): boolean {
  return supported === null || supported.includes(id);
}
