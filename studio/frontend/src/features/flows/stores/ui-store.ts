// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { EdgeKind, PortSpec } from "../types";

export interface QuickAddState {
  /** Screen position the popover opens at. */
  screen: { x: number; y: number };
  /** Flow position a new node lands at. */
  flow: { x: number; y: number };
  /** Set when the user dropped a dangling connection: the new node is auto-connected. */
  from?: {
    nodeId: string;
    handleId: string;
    side: "in" | "out";
    kind: EdgeKind;
    port: PortSpec | null;
  };
}

export type SaveState = "saved" | "dirty" | "saving" | "error" | "conflict";

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
  setSelection: (nodes: string[], edges: string[]) => void;
  openInspector: (nodeId: string | null) => void;
  setQuickAdd: (quickAdd: QuickAddState | null) => void;
  setDrawer: (open: boolean, tab?: UiState["drawerTab"]) => void;
  togglePalette: () => void;
  toggleSnap: () => void;
  setSaveState: (state: SaveState) => void;
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
    }),
}));
