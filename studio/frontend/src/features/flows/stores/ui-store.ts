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

/** Bus: routed orthogonal wires with rounded corners and lanes. Curved: plain bezier wires. */
export type WireStyle = "bus" | "curved";

export const WIRE_STYLE_STORAGE_KEY = "unsloth_flows_wire_style";

export function storedWireStyle(): WireStyle {
  try {
    return globalThis.localStorage?.getItem(WIRE_STYLE_STORAGE_KEY) === "curved"
      ? "curved"
      : "bus";
  } catch {
    return "bus";
  }
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
  wireStyle: WireStyle;
  saveState: SaveState;
  pendingRun: PendingRun | null;
  setSelection: (nodes: string[], edges: string[]) => void;
  openInspector: (nodeId: string | null) => void;
  setQuickAdd: (quickAdd: QuickAddState | null) => void;
  setDrawer: (open: boolean, tab?: UiState["drawerTab"]) => void;
  togglePalette: () => void;
  toggleSnap: () => void;
  setWireStyle: (style: WireStyle) => void;
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
  wireStyle: storedWireStyle(),
  saveState: "saved",
  pendingRun: null,
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
  setWireStyle: (wireStyle) => {
    set({ wireStyle });
    try {
      globalThis.localStorage?.setItem(WIRE_STYLE_STORAGE_KEY, wireStyle);
    } catch {
      // Private mode or no storage: the choice lasts until reload.
    }
  },
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
    }),
}));
