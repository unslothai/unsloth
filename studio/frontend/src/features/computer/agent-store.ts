// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { ComputerAction, ComputerPermissionMode } from "./types";

export interface ComputerAgentState {
  enabled: boolean;
  permissionMode: ComputerPermissionMode;
  activeAction: ComputerAction | null;
  cursorPosition: { x: number; y: number } | null;
  intervalCaptureEnabled: boolean;
  intervalSeconds: number;
  abortRequested: boolean;

  setEnabled: (enabled: boolean) => void;
  setPermissionMode: (mode: ComputerPermissionMode) => void;
  setActiveAction: (action: ComputerAction | null) => void;
  setCursorPosition: (pos: { x: number; y: number } | null) => void;
  setIntervalCapture: (enabled: boolean, intervalSeconds?: number) => void;
  abortControl: () => void;
  resetAbort: () => void;
}

export const useComputerAgentStore = create<ComputerAgentState>((set) => ({
  enabled: false,
  permissionMode: "restricted",
  activeAction: null,
  cursorPosition: null,
  intervalCaptureEnabled: false,
  intervalSeconds: 5,
  abortRequested: false,

  setEnabled: (enabled) => set({ enabled }),
  setPermissionMode: (permissionMode) => set({ permissionMode }),
  setActiveAction: (activeAction) => set({ activeAction }),
  setCursorPosition: (cursorPosition) => set({ cursorPosition }),
  setIntervalCapture: (enabled, intervalSeconds) =>
    set((state) => ({
      intervalCaptureEnabled: enabled,
      intervalSeconds: intervalSeconds ?? state.intervalSeconds,
    })),
  abortControl: () => set({ abortRequested: true, activeAction: null }),
  resetAbort: () => set({ abortRequested: false }),
}));
