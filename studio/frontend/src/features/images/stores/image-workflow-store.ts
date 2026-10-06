// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { WorkflowId } from "../workflows";

/** The Images page's active workflow, lifted out of the page so the sidebar submenu can drive it.
 *  `supported` is what the loaded model can do, published by the page; null means nothing is
 *  loaded, in which case every workflow stays selectable. */
interface ImageWorkflowState {
  workflow: WorkflowId;
  pageMode: "create" | "train";
  supported: WorkflowId[] | null;
  /** Off the Images page, whether the sidebar lists the workflows under the row. */
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
  // Picking a workflow implies Create: workflows do not exist in Train.
  setWorkflow: (workflow) => set({ workflow, pageMode: "create" }),
  setPageMode: (pageMode) => set({ pageMode }),
  setSupported: (supported) => set({ supported }),
}));

/** Only a loaded model closes a workflow off, and only one it cannot do. */
export function isWorkflowEnabled(
  id: WorkflowId,
  supported: WorkflowId[] | null,
): boolean {
  return supported === null || supported.includes(id);
}
