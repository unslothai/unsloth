// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { useBenchmarksStore } from "@/features/benchmarks/stores/benchmarks-store";
import {
  cancelBenchmark,
  runBenchmark,
  getBenchmarkStatus,
  isRecoverableTransportError,
  type BenchmarkLogEntry,
  type BenchmarkOperationResponse,
  type BenchmarkStatus,
} from "../api/benchmark-api";

class BenchmarkCanceledError extends Error {
  constructor() {
    super("Benchmark canceled");
    this.name = "BenchmarkCanceledError";
  }
}

const RECOVERY_POLL_INTERVAL_MS = 1500;
const RECOVERY_GRACE_MS = 15000;
const RECOVERY_MAX_MS = 2 * 60 * 60 * 1000;
const RECOVERY_MAX_STATUS_FAILS = 5;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function recoverViaStatus(
  baseline: number | null,
  isCurrent: () => boolean,
): Promise<void> {
  const start = Date.now();
  let statusFails = 0;
  let sawActive = false;

  while (Date.now() - start < RECOVERY_MAX_MS) {
    if (!isCurrent()) throw new Error("Benchmark run superseded");
    await sleep(RECOVERY_POLL_INTERVAL_MS);

    let st: BenchmarkStatus;
    try {
      st = await getBenchmarkStatus();
      statusFails = 0;
    } catch {
      statusFails += 1;
      if (
        Date.now() - start > RECOVERY_GRACE_MS &&
        statusFails >= RECOVERY_MAX_STATUS_FAILS
      ) {
        throw new Error("Lost connection to the benchmark server.");
      }
      continue;
    }

    if (st.is_benchmark_active) {
      sawActive = true;
      continue;
    }

    const seq = st.last_op_seq ?? 0;
    const isOurs = sawActive || (baseline !== null && seq > baseline);
    if (isOurs) {
      if (st.last_op_status === "success") return;
      if (st.last_op_status === "cancelled") throw new BenchmarkCanceledError();
      throw new Error(st.last_op_error || "Benchmark failed");
    }

    if (Date.now() - start > RECOVERY_GRACE_MS) {
      throw new Error(
        "The benchmark request failed before the server started the operation.",
      );
    }
  }
  throw new Error("Timed out waiting for benchmark to finish.");
}

const MAX_LOG_LINES = 4000;

export type BenchmarkPhase =
  | "idle"
  | "starting"
  | "running"
  | "success"
  | "error"
  | "canceled";

export interface BenchmarkProgress {
  pct: number;
  current: number;
  total: number;
  /** Formatted by the backend's ``tqdm.format_interval`` — null before a rate is known. */
  elapsed: string | null;
  eta: string | null;
}

interface BenchmarkRuntimeState {
  phase: BenchmarkPhase;
  isRunning: boolean;
  ownsRun: boolean;
  logLines: BenchmarkLogEntry[];
  lastSeq: number | null;
  connected: boolean;
  reconnecting: boolean;
  startedAt: number | null;
  stage: string | null;
  progressPercent: number;
  progressDetail: BenchmarkProgress | null;
  result: string | null;
  error: string | null;
  cancelRequested: boolean;
  hasHydrated: boolean;
  backendActive: boolean;
  runId: number;
}

interface BenchmarkRuntimeActions {
  run: (checkpointPath: string, modelSource: "checkpoint" | "hf" | "local", task?: string, extraParams?: { batch_size?: string; num_fewshot?: number | null; max_tokens?: number | null }) => Promise<void>;
  requestCancel: () => Promise<void>;
  appendLog: (entry: BenchmarkLogEntry, seq?: number) => void;
  setConnected: (value: boolean) => void;
  applyBackendStatus: (status: BenchmarkStatus) => void;
  reset: () => void;
}

export type BenchmarkRuntimeStore = BenchmarkRuntimeState & BenchmarkRuntimeActions;

const initialState: BenchmarkRuntimeState = {
  phase: "idle",
  isRunning: false,
  ownsRun: false,
  logLines: [],
  lastSeq: null,
  connected: false,
  reconnecting: false,
  startedAt: null,
  stage: null,
  progressPercent: 0,
  progressDetail: null,
  result: null,
  error: null,
  cancelRequested: false,
  hasHydrated: false,
  backendActive: false,
  runId: 0,
};

function _parseProgress(line: string): { pct: number; detail: BenchmarkProgress | null } {
  try {
    const parsed = JSON.parse(line);
    return { pct: parsed.pct, detail: parsed };
  } catch {
    const pct = Number(line);
    return { pct: Number.isFinite(pct) ? pct : 0, detail: null };
  }
}

export const useBenchmarkRuntimeStore = create<BenchmarkRuntimeStore>()((set, get) => ({
  ...initialState,

  setConnected: (value) => set({ connected: value }),

  appendLog: (entry, seq) =>
    set((state) => {
      if (
        typeof seq === "number" &&
        state.lastSeq !== null &&
        seq <= state.lastSeq
      ) {
        return state;
      }

      let progressPercent = state.progressPercent;
      let progressDetail: BenchmarkProgress | null = state.progressDetail;
      if (entry.stream === "progress") {
        const parsed = _parseProgress(entry.line);
        progressPercent = parsed.pct;
        progressDetail = parsed.detail;
      }

      const next =
        state.logLines.length >= MAX_LOG_LINES
          ? state.logLines.slice(state.logLines.length - MAX_LOG_LINES + 1)
          : state.logLines.slice();
      next.push(entry);
      return {
        logLines: next,
        lastSeq: typeof seq === "number" ? seq : state.lastSeq,
        stage: entry.stream === "status" ? entry.line : state.stage,
        progressPercent,
        progressDetail,
      };
    }),

  applyBackendStatus: (status) =>
    set((state) => {
      const base = { hasHydrated: true, backendActive: status.is_benchmark_active };
      if (status.is_benchmark_active && !state.isRunning && !state.ownsRun) {
        return {
          ...base,
          isRunning: true,
          phase: "running" as const,
          startedAt: state.startedAt ?? Date.now(),
        };
      }
      if (!status.is_benchmark_active && state.isRunning && !state.ownsRun) {
        if (status.last_op_status === "error") {
          return {
            ...base,
            isRunning: false,
            phase: "error" as const,
            error: status.last_op_error ?? "Benchmark failed",
          };
        }
        if (status.last_op_status === "cancelled") {
          return { ...base, isRunning: false, phase: "canceled" as const };
        }
        if (status.last_op_status === "success") {
          return {
            ...base,
            isRunning: false,
            phase: "success" as const,
          };
        }
        return { ...base, isRunning: false, phase: "idle" as const };
      }
      return base;
    }),

  reset: () =>
    set((state) => ({
      ...initialState,
      hasHydrated: state.hasHydrated,
      backendActive: state.backendActive,
      runId: state.runId,
    })),

  requestCancel: async () => {
    if (!get().isRunning) return;
    set({ cancelRequested: true });
    try {
      await cancelBenchmark();
    } catch {
      // the stream/status poll will observe the terminal state instead
    }
  },

  run: async (checkpointPath, modelSource, task, extraParams) => {
    // A Config sweep reloads the server between rows; scoring under it is meaningless.
    if (useBenchmarksStore.getState().live) {
      set({
        phase: "error",
        error: "A Config sweep is running. Stop it or let it finish first.",
      });
      return;
    }
    const runId = get().runId + 1;

    set({
      runId,
      isRunning: true,
      ownsRun: true,
      phase: "starting",
      stage: null,
      progressPercent: 0,
      progressDetail: null,
      logLines: [],
      lastSeq: null,
      connected: false,
      reconnecting: false,
      startedAt: Date.now(),
      result: null,
      error: null,
      cancelRequested: false,
    });

    const isCurrent = () => get().runId === runId;

    const runRecoverableOp = async (
      post: () => Promise<BenchmarkOperationResponse>,
    ): Promise<void> => {
      let baseline: number | null = null;
      try {
        baseline = (await getBenchmarkStatus()).last_op_seq ?? 0;
      } catch {
        baseline = null;
      }
      let res: BenchmarkOperationResponse;
      try {
        res = await post();
      } catch (err) {
        if (!isRecoverableTransportError(err)) throw err;
        set({ reconnecting: true });
        try {
          await recoverViaStatus(baseline, isCurrent);
        } finally {
          if (isCurrent()) set({ reconnecting: false });
        }
        return;
      }
      // A run stopped mid-way answers 200 with success=false.
      if (res.success === false) {
        if (res.details?.cancelled) throw new BenchmarkCanceledError();
        throw new Error(res.message || "Benchmark failed");
      }
    };

    try {
const apiParams = {
          checkpoint_path: checkpointPath,
          model_source: modelSource,
          task: task ?? "mmlu",
          batch_size: extraParams?.batch_size ?? "auto",
          num_fewshot: extraParams?.num_fewshot ?? null,
          max_tokens: extraParams?.max_tokens ?? null,
        };
      await runRecoverableOp(() =>
        runBenchmark(apiParams),
      );
      if (!isCurrent()) return;

      set({
        phase: "running",
      });

      if (!isCurrent()) return;

      set({
        phase: "success",
        isRunning: false,
        reconnecting: false,
        result: "Benchmark completed",
      });
    } catch (err) {
      if (!isCurrent()) return;
      if (get().cancelRequested || err instanceof BenchmarkCanceledError) {
        set({
          phase: "canceled",
          isRunning: false,
          reconnecting: false,
          error: null,
        });
      } else {
        set({
          phase: "error",
          isRunning: false,
          reconnecting: false,
          error: err instanceof Error ? err.message : "Benchmark failed",
        });
      }
    }
  },
}));

export function isBenchmarkPanelActive(state: BenchmarkRuntimeStore): boolean {
  return state.isRunning || state.phase !== "idle";
}

export function selectBenchmarkProgressPercent(state: BenchmarkRuntimeStore): number {
  if (state.phase === "idle") return 0;
  if (state.phase === "success") return 100;
  return state.progressPercent;
}

