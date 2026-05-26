// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { EvalProgress, EvalStatus } from "../api/eval-api";

export interface EvalMiniResult {
  idx: number;
  score: number;
  error?: string | null;
}

interface EvalRuntimeState {
  currentRunId: string | null;
  status: EvalStatus | "idle";
  done: number;
  total: number;
  avgScore: number;
  startedAtMs: number | null;
  isEvalRunning: boolean;
  liveResults: EvalMiniResult[]; // accumulated from SSE last_result, deduped by idx
  startError: string | null;
  selectedHistoryRunId: string | null;

  beginRun: (runId: string, total: number) => void;
  applyProgress: (p: EvalProgress) => void;
  finishRun: (status: EvalStatus) => void;
  setStartError: (msg: string | null) => void;
  setSelectedHistoryRunId: (id: string | null) => void;
  resetRuntime: () => void;
}

const initial = {
  currentRunId: null as string | null,
  status: "idle" as EvalStatus | "idle",
  done: 0,
  total: 0,
  avgScore: 0,
  startedAtMs: null as number | null,
  isEvalRunning: false,
  liveResults: [] as EvalMiniResult[],
  startError: null as string | null,
};

export const useEvalRuntimeStore = create<EvalRuntimeState>()((set) => ({
  ...initial,
  selectedHistoryRunId: null,

  beginRun: (runId, total) =>
    set({
      currentRunId: runId,
      status: "running",
      done: 0,
      total,
      avgScore: 0,
      startedAtMs: Date.now(),
      isEvalRunning: true,
      liveResults: [],
      startError: null,
      selectedHistoryRunId: runId,
    }),

  applyProgress: (p) =>
    set((s) => {
      const liveResults = s.liveResults.slice();
      if (p.last_result) {
        const i = liveResults.findIndex((r) => r.idx === p.last_result!.idx);
        const entry: EvalMiniResult = {
          idx: p.last_result.idx,
          score: p.last_result.score,
          error: p.last_result.error,
        };
        if (i >= 0) liveResults[i] = entry;
        else liveResults.push(entry);
      }
      return {
        status: p.status,
        done: p.done,
        total: p.total || s.total,
        avgScore: p.avg_score,
        isEvalRunning: p.status === "running",
        liveResults,
      };
    }),

  finishRun: (status) => set({ status, isEvalRunning: false }),
  setStartError: (msg) => set({ startError: msg }),
  setSelectedHistoryRunId: (id) => set({ selectedHistoryRunId: id }),
  resetRuntime: () => set({ ...initial }),
}));
