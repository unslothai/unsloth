// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  cancelExport,
  cleanupExport,
  exportBase,
  exportGGUF,
  exportLoRA,
  exportMerged,
  loadCheckpoint,
  type ExportLogEntry,
  type ExportStatus,
} from "../api/export-api";
import type { ExportMethod } from "../constants";

// Keep the same scrollback depth as the backend ring buffer so the inline
// panel shows the full server-side history.
const MAX_LOG_LINES = 4000;

export type ExportPhase =
  | "idle"
  | "loading"
  | "exporting"
  | "success"
  | "error"
  | "canceled";

export type ExportDestination = "local" | "hub";

/** Snapshot of what is being exported, captured when a run starts. */
export interface ExportRunSummary {
  baseModelName: string;
  checkpointLabel: string | null;
  methodLabel: string;
  method: ExportMethod;
  quantLevels: string[];
  destination: ExportDestination;
}

/** Everything `runExport` needs to drive the load -> export -> cleanup sequence. */
export interface RunExportParams {
  sourceMode: "checkpoint" | "model";
  /** Resolved on-disk checkpoint path (checkpoint mode). */
  checkpointPath: string | null;
  /** Raw source id passed to load-checkpoint (model mode: HF id or local path). */
  source: string;
  modelSource: "hf" | "local";
  trustRemoteCode: boolean;
  exportMethod: ExportMethod;
  isAdapter: boolean;
  quantLevels: string[];
  saveDirectory: string;
  destination: ExportDestination;
  repoId?: string;
  token?: string;
  privateRepo: boolean;
  baseModelId?: string | null;
  summary: ExportRunSummary;
}

interface ExportRuntimeState {
  phase: ExportPhase;
  /** True from the moment a run starts until it reaches a terminal phase. */
  isExporting: boolean;
  /** True while this store's own `runExport` drives the sequence (vs a run
   *  recovered from the backend after a reload). Gates status-poll takeovers. */
  ownsRun: boolean;
  method: ExportMethod | null;
  summary: ExportRunSummary | null;
  quantTotal: number;
  /** Number of GGUF quants finished so far (0-based current = this value). */
  quantIndex: number;
  /** Latest high-level status line surfaced by the worker. */
  stage: string | null;
  logLines: ExportLogEntry[];
  lastSeq: number | null;
  connected: boolean;
  startedAt: number | null;
  result: { outputPath: string | null; destination: ExportDestination } | null;
  error: string | null;
  cancelRequested: boolean;
  hasHydrated: boolean;
  backendActive: boolean;
  /** Bumped on every run so stale async callbacks can detect they were superseded. */
  runId: number;
}

interface ExportRuntimeActions {
  runExport: (params: RunExportParams) => Promise<void>;
  requestCancel: () => Promise<void>;
  appendLog: (entry: ExportLogEntry, seq?: number) => void;
  setConnected: (value: boolean) => void;
  applyBackendStatus: (status: ExportStatus) => void;
  reset: () => void;
}

export type ExportRuntimeStore = ExportRuntimeState & ExportRuntimeActions;

const initialState: ExportRuntimeState = {
  phase: "idle",
  isExporting: false,
  ownsRun: false,
  method: null,
  summary: null,
  quantTotal: 1,
  quantIndex: 0,
  stage: null,
  logLines: [],
  lastSeq: null,
  connected: false,
  startedAt: null,
  result: null,
  error: null,
  cancelRequested: false,
  hasHydrated: false,
  backendActive: false,
  runId: 0,
};

export const useExportRuntimeStore = create<ExportRuntimeStore>()((set, get) => ({
  ...initialState,

  setConnected: (value) => set({ connected: value }),

  appendLog: (entry, seq) =>
    set((state) => {
      const next =
        state.logLines.length >= MAX_LOG_LINES
          ? state.logLines.slice(state.logLines.length - MAX_LOG_LINES + 1)
          : state.logLines.slice();
      next.push(entry);
      return {
        logLines: next,
        lastSeq: typeof seq === "number" ? seq : state.lastSeq,
        // `status` lines are the worker's high-level progress markers; surface
        // the most recent one as the stage label.
        stage: entry.stream === "status" ? entry.line : state.stage,
      };
    }),

  applyBackendStatus: (status) =>
    set((state) => {
      const base = { hasHydrated: true, backendActive: status.is_export_active };
      // Recover a run started before this store existed (full page reload, or
      // an export kicked off in another browser tab): show it live.
      if (status.is_export_active && !state.isExporting && !state.ownsRun) {
        return {
          ...base,
          isExporting: true,
          phase: "exporting" as const,
          startedAt: state.startedAt ?? Date.now(),
        };
      }
      // A recovered (not store-owned) run finished on the backend. We can't know
      // success vs failure for a run we didn't drive, so settle optimistically.
      if (!status.is_export_active && state.isExporting && !state.ownsRun) {
        return {
          ...base,
          isExporting: false,
          phase: state.logLines.length > 0 ? ("success" as const) : ("idle" as const),
        };
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
    if (!get().isExporting) return;
    set({ cancelRequested: true });
    try {
      await cancelExport();
    } catch {
      // Best-effort: the in-flight export POST will still reject when the
      // worker dies, which runExport turns into the canceled phase.
    }
  },

  runExport: async (params) => {
    const runId = get().runId + 1;
    const quantTotal =
      params.exportMethod === "gguf"
        ? Math.max(1, params.quantLevels.length)
        : 1;

    set({
      runId,
      isExporting: true,
      ownsRun: true,
      phase: "loading",
      method: params.exportMethod,
      summary: params.summary,
      quantTotal,
      quantIndex: 0,
      stage: null,
      logLines: [],
      lastSeq: null,
      connected: false,
      startedAt: Date.now(),
      result: null,
      error: null,
      cancelRequested: false,
    });

    const isCurrent = () => get().runId === runId;
    const pushToHub = params.destination === "hub";

    try {
      // 1. Load the model source into a fresh export subprocess.
      if (params.sourceMode === "checkpoint") {
        if (!params.checkpointPath) {
          throw new Error("No checkpoint selected");
        }
        await loadCheckpoint({ checkpoint_path: params.checkpointPath });
      } else {
        await loadCheckpoint({
          checkpoint_path: params.source,
          load_in_4bit: false,
          trust_remote_code:
            params.modelSource === "hf" ? params.trustRemoteCode : true,
        });
      }
      if (!isCurrent()) return;

      // 2. Run the export. Capture the resolved output_path for the success
      // banner; multi-quant GGUF shares one directory, so keep the last.
      set({ phase: "exporting" });
      let lastOutputPath: string | null = null;

      if (params.exportMethod === "merged") {
        if (params.isAdapter) {
          const resp = await exportMerged({
            save_directory: params.saveDirectory,
            push_to_hub: pushToHub,
            repo_id: params.repoId,
            hf_token: params.token,
            private: params.privateRepo,
          });
          lastOutputPath = resp.details?.output_path ?? null;
        } else {
          const resp = await exportBase({
            save_directory: params.saveDirectory,
            push_to_hub: pushToHub,
            repo_id: params.repoId,
            hf_token: params.token,
            private: params.privateRepo,
            base_model_id: params.baseModelId,
          });
          lastOutputPath = resp.details?.output_path ?? null;
        }
      } else if (params.exportMethod === "gguf") {
        for (let i = 0; i < params.quantLevels.length; i += 1) {
          if (!isCurrent()) return;
          set({ quantIndex: i });
          const resp = await exportGGUF({
            save_directory: params.saveDirectory,
            quantization_method: params.quantLevels[i],
            push_to_hub: pushToHub,
            repo_id: params.repoId,
            hf_token: params.token,
          });
          lastOutputPath = resp.details?.output_path ?? lastOutputPath;
          if (!isCurrent()) return;
          set({ quantIndex: i + 1 });
        }
      } else if (params.exportMethod === "lora") {
        const resp = await exportLoRA({
          save_directory: params.saveDirectory,
          push_to_hub: pushToHub,
          repo_id: params.repoId,
          hf_token: params.token,
          private: params.privateRepo,
        });
        lastOutputPath = resp.details?.output_path ?? null;
      }
      if (!isCurrent()) return;

      set({
        phase: "success",
        isExporting: false,
        result: { outputPath: lastOutputPath, destination: params.destination },
      });
    } catch (err) {
      if (!isCurrent()) return;
      if (get().cancelRequested) {
        set({ phase: "canceled", isExporting: false, error: null });
      } else {
        set({
          phase: "error",
          isExporting: false,
          error: err instanceof Error ? err.message : "Export failed",
        });
      }
    } finally {
      // Cleanup is best-effort and runs after the terminal phase is set, so it
      // does not gate the success banner. Only the run that still owns the
      // store releases ownership and frees the worker.
      if (isCurrent()) {
        try {
          await cleanupExport();
        } catch {
          // ignore
        }
        if (isCurrent()) {
          set({ ownsRun: false });
        }
      }
    }
  },
}));

/**
 * Map the run phase to a 0-100 progress value. There is no byte-level signal
 * from llama.cpp / the HF uploader, so progress is phase + quant-index based:
 * loading occupies a small head band, the export body advances per completed
 * GGUF quant, and success pins to 100.
 */
export function selectExportProgressPercent(state: ExportRuntimeStore): number {
  const total = Math.max(1, state.quantTotal);
  const exportBand = () => {
    const done = Math.min(Math.max(state.quantIndex, 0), total);
    return Math.round(15 + (done / total) * 72); // 15..87
  };
  switch (state.phase) {
    case "idle":
      return 0;
    case "loading":
      return 8;
    case "exporting":
      return exportBand();
    case "success":
      return 100;
    case "error":
    case "canceled":
      // Freeze near where it stopped so the bar does not snap back to 0.
      return Math.max(8, exportBand());
    default:
      return 0;
  }
}

/** Whether the inline run panel should be visible (a run is active or terminal). */
export function isExportPanelActive(state: ExportRuntimeStore): boolean {
  return state.isExporting || state.phase !== "idle";
}
