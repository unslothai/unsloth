// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { FlowPlan } from "../api";
import type { EdgeKind, PortSpec } from "../types";

export interface QuickAddState {
  screen: { x: number; y: number };
  flow: { x: number; y: number };
  /** a recipe group: list the blocks it can take instead of nodes */
  recipe?: string;
  from?: {
    nodeId: string;
    handleId: string;
    side: "in" | "out";
    kind: EdgeKind;
    port: PortSpec | null;
  };
}

/** A Recipe Studio dialog opened over a recipe group: its run settings or the GitHub crawler form. */
export interface RecipeEditorTarget {
  groupId: string;
  runSettings?: boolean;
  /** Recipe Studio's GitHub crawler form, for a GitHub-seeded recipe */
  easy?: boolean;
}

/** A recipe group's seed, provider connection, model preset or tool profile, open in the inspector. */
export interface RecipeSetupTarget {
  groupId: string;
  kind: "seed" | "model_provider" | "model_config" | "tool_config";
  name: string;
}

/** A recipe group's runs, in Recipe Studio's executions view: the newest one only, or all of them. */
export interface RecipeRunsTarget {
  groupId: string;
  view: "preview" | "runs";
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
  recipeSetup: RecipeSetupTarget | null;
  /** bumps when a different setup card opens, not when the open one is renamed */
  recipeSetupKey: number;
  setRecipeSetup: (target: RecipeSetupTarget | null, renamed?: boolean) => void;
  recipeRuns: RecipeRunsTarget | null;
  setRecipeRuns: (target: RecipeRunsTarget | null) => void;
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
  recipeSetup: null,
  recipeSetupKey: 0,
  setRecipeSetup: (recipeSetup, renamed = false) =>
    set((s) => ({
      recipeSetup,
      recipeSetupKey: renamed ? s.recipeSetupKey : s.recipeSetupKey + 1,
      ...(recipeSetup && { inspectorNodeId: null }),
    })),
  recipeRuns: null,
  setRecipeRuns: (recipeRuns) => set({ recipeRuns }),
  setSelection: (nodes, edges) => {
    const s = get();
    if (sameList(s.selectedNodes, nodes) && sameList(s.selectedEdges, edges))
      return;
    set({
      selectedNodes: nodes,
      selectedEdges: edges,
      ...(nodes.length === 1 && { recipeSetup: null }),
      inspectorNodeId:
        nodes.length === 1
          ? nodes[0]
          : nodes.length === 0
            ? s.inspectorNodeId
            : s.inspectorNodeId,
    });
  },
  openInspector: (nodeId) =>
    set(
      nodeId
        ? { inspectorNodeId: nodeId, recipeSetup: null }
        : { inspectorNodeId: nodeId },
    ),
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
      recipeSetup: null,
      recipeRuns: null,
    }),
}));
