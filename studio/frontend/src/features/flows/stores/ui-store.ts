// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { FlowPlan } from "../api";
import type { EdgeKind, PortSpec } from "../types";

export interface QuickAddState {
  screen: { x: number; y: number };
  flow: { x: number; y: number };
  from?: {
    nodeId: string;
    handleId: string;
    side: "in" | "out";
    kind: EdgeKind;
    port: PortSpec | null;
  };
}

/** What the Recipe Studio editor opened over a recipe group edits: a column, a recipe-level block, a new
 *  block from Recipe Studio's catalog, or the processors. */
export interface RecipeEditorTarget {
  groupId: string;
  nodeId?: string;
  configName?: string;
  seed?: boolean;
  add?: { kind: string; type: string };
  processors?: boolean;
}

export type SaveState = "saved" | "dirty" | "saving" | "error" | "conflict";

export interface PendingRun {
  options: Record<string, unknown>;
  plan: FlowPlan;
  models?: string;
}

export interface UiState {
  selectedNodes: string[];
  selectedEdges: string[];
  inspectorNodeId: string | null;
  quickAdd: QuickAddState | null;
  drawerOpen: boolean;
  drawerTab: "run" | "logs" | "outputs";
  paletteOpen: boolean;
  snapToGrid: boolean;
  saveState: SaveState;
  pendingRun: PendingRun | null;
  recipeEditor: RecipeEditorTarget | null;
  setRecipeEditor: (target: RecipeEditorTarget | null) => void;
  setSelection: (nodes: string[], edges: string[]) => void;
  openInspector: (nodeId: string | null) => void;
  setQuickAdd: (quickAdd: QuickAddState | null) => void;
  setDrawer: (open: boolean, tab?: UiState["drawerTab"]) => void;
  togglePalette: () => void;
  toggleSnap: () => void;
  setSaveState: (state: SaveState) => void;
  setPendingRun: (pending: PendingRun | null) => void;
  reset: () => void;
}

const sameList = (a: string[], b: string[]) =>
  a.length === b.length && a.every((v, i) => v === b[i]);

export const useUiStore = create<UiState>((set, get) => ({
  selectedNodes: [],
  selectedEdges: [],
  inspectorNodeId: null,
  quickAdd: null,
  drawerOpen: false,
  drawerTab: "run",
  paletteOpen: true,
  snapToGrid: false,
  saveState: "saved",
  pendingRun: null,
  recipeEditor: null,
  setRecipeEditor: (recipeEditor) => set({ recipeEditor }),
  setSelection: (nodes, edges) => {
    const s = get();
    if (sameList(s.selectedNodes, nodes) && sameList(s.selectedEdges, edges))
      return;
    set({
      selectedNodes: nodes,
      selectedEdges: edges,
      inspectorNodeId:
        nodes.length === 1
          ? nodes[0]
          : nodes.length === 0
            ? s.inspectorNodeId
            : s.inspectorNodeId,
    });
  },
  openInspector: (nodeId) => set({ inspectorNodeId: nodeId }),
  setQuickAdd: (quickAdd) => set({ quickAdd }),
  setDrawer: (open, tab) =>
    set((s) => ({ drawerOpen: open, drawerTab: tab ?? s.drawerTab })),
  togglePalette: () => set((s) => ({ paletteOpen: !s.paletteOpen })),
  toggleSnap: () => set((s) => ({ snapToGrid: !s.snapToGrid })),
  setPendingRun: (pendingRun) => set({ pendingRun }),
  setSaveState: (saveState) => {
    if (get().saveState !== saveState) set({ saveState });
  },
  reset: () =>
    set({
      selectedNodes: [],
      selectedEdges: [],
      inspectorNodeId: null,
      quickAdd: null,
      saveState: "saved",
      pendingRun: null,
      recipeEditor: null,
    }),
}));
