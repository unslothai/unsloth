// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  cancelExport,
  cleanupExport,
  exportGGUF,
  exportLoRA,
  exportMerged,
  getExportStatus,
  isRecoverableTransportError,
  loadCheckpoint,
  type ExportLogEntry,
  type ExportLogPollEntry,
  type ExportOperationResponse,
  type ExportStatus,
} from "../api/export-api";
import type { ExportMethod } from "../constants";

class ExportCanceledError extends Error {
  constructor() {
    super("Export canceled");
    this.name = "ExportCanceledError";
  }
}

// Recovery when a Cloudflare 524 cuts off a blocking export POST while the op keeps running.
const RECOVERY_POLL_INTERVAL_MS = 1500;
const RECOVERY_GRACE_MS = 15000; // wait this long for the op to appear on status
const RECOVERY_MAX_MS = 2 * 60 * 60 * 1000;
const RECOVERY_MAX_STATUS_FAILS = 5;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Settle a phase whose POST hit a tunnel timeout by polling status. A record is ours once
 *  the op is inactive and we saw it active or its seq passed `baseline`. */
async function recoverViaStatus(
  baseline: number | null,
  isCurrent: () => boolean,
): Promise<{ outputPath: string | null }> {
  const start = Date.now();
  let statusFails = 0;
  let sawActive = false;

  while (Date.now() - start < RECOVERY_MAX_MS) {
    if (!isCurrent()) throw new Error("Export run superseded");
    await sleep(RECOVERY_POLL_INTERVAL_MS);

    let st: ExportStatus;
    try {
      st = await getExportStatus();
      statusFails = 0;
    } catch {
      statusFails += 1;
      if (
        Date.now() - start > RECOVERY_GRACE_MS &&
        statusFails >= RECOVERY_MAX_STATUS_FAILS
      ) {
        throw new Error("Lost connection to the export server.");
      }
      continue;
    }

    if (st.is_export_active) {
      sawActive = true;
      continue;
    }

    const seq = st.last_op_seq ?? 0;
    const isOurs = sawActive || (baseline !== null && seq > baseline);
    if (isOurs) {
      if (st.last_op_status === "success") {
        return { outputPath: st.last_op_output_path ?? null };
      }
      if (st.last_op_status === "cancelled") throw new ExportCanceledError();
      throw new Error(st.last_op_error || "Export failed");
    }

    // Our op was never observed: the POST likely died first. Fail after a grace window.
    if (Date.now() - start > RECOVERY_GRACE_MS) {
      throw new Error(
        "The export request failed before the server started the operation.",
      );
    }
  }
  throw new Error("Timed out waiting for the export to finish.");
}

/** Every quant in the run folder's gguf/export.json, else the ones requested. */
export function decisionQuantizations(
  details: ExportOperationResponse["details"] | undefined,
  requested: string[],
): string[] {
  const reported = details?.quantizations;
  return Array.isArray(reported) && reported.length > 0
    ? reported.map(String)
    : requested.map((q) => q.toUpperCase());
}

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

export interface ExportRunSummary {
  baseModelName: string;
  checkpointLabel: string | null;
  methodLabel: string;
  method: ExportMethod;
  quantLevels: string[];
  mergedFormats: string[];
  destination: ExportDestination;
}

export interface RunExportParams {
  sourceMode: "checkpoint" | "model";
  checkpointPath: string | null;
  source: string;
  modelSource: "hf" | "local";
  trustRemoteCode: boolean;
  approvedRemoteCodeFingerprint?: string | null;
  loadToken?: string | null;
  exportMethod: ExportMethod;
  isAdapter: boolean;
  quantLevels: string[];
  useImatrix?: boolean;
  imatrixPath?: string;
  npuQ4nx?: boolean;
  mergedSelections?: {
    formatType: string;
    compressedMethod: string | null;
    label: string;
  }[];
  loraGguf?: boolean;
  adapterFormat?: "mlx" | "peft";
  loraGgufOuttype?: string;
  saveDirectory: string;
  destination: ExportDestination;
  repoId?: string;
  token?: string;
  privateRepo: boolean;
  baseModelId?: string | null;
  installMissingDependencies?: boolean;
  /** Decision model (Clef / Laya): labels the run folder's gguf/ output with the quants it holds. */
  decisionOutputLabel?: (quantizations: string[]) => string;
  summary: ExportRunSummary;
}

interface ExportRuntimeState {
  phase: ExportPhase;
  isExporting: boolean;
  /** True while this store's runExport drives the run (not a reload-recovered one). */
  ownsRun: boolean;
  method: ExportMethod | null;
  summary: ExportRunSummary | null;
  quantTotal: number;
  quantIndex: number;
  stage: string | null;
  logLines: ExportLogEntry[];
  lastSeq: number | null;
  connected: boolean;
  /** POST response lost to a tunnel timeout; settling via /api/export/status. */
  reconnecting: boolean;
  startedAt: number | null;
  /** `outputPath` is kept for back-compat; `outputPaths` has one entry per folder. */
  result: {
    outputPath: string | null;
    outputPaths: { label: string; path: string }[];
    destination: ExportDestination;
  } | null;
  error: string | null;
  cancelRequested: boolean;
  hasHydrated: boolean;
  backendActive: boolean;
  runId: number;
}

interface ExportRuntimeActions {
  runExport: (params: RunExportParams) => Promise<void>;
  requestCancel: () => Promise<void>;
  appendLog: (entry: ExportLogEntry, seq?: number) => void;
  appendLogs: (entries: ExportLogPollEntry[]) => void;
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
  reconnecting: false,
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
      // SSE and the poll fallback both feed logs, so dedupe by seq.
      if (
        typeof seq === "number" &&
        state.lastSeq !== null &&
        seq <= state.lastSeq
      ) {
        return state;
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
      };
    }),

  appendLogs: (entries) =>
    set((state) => {
      const fresh =
        state.lastSeq === null
          ? entries
          : entries.filter((e) => e.seq > (state.lastSeq as number));
      if (fresh.length === 0) return state;

      const merged = state.logLines.concat(
        fresh.map((e) => ({ stream: e.stream, line: e.line, ts: e.ts })),
      );
      const next =
        merged.length > MAX_LOG_LINES
          ? merged.slice(merged.length - MAX_LOG_LINES)
          : merged;

      let stage = state.stage;
      for (const e of fresh) {
        if (e.stream === "status") stage = e.line;
      }
      return {
        logLines: next,
        lastSeq: fresh[fresh.length - 1].seq,
        stage,
      };
    }),

  applyBackendStatus: (status) =>
    set((state) => {
      const base = { hasHydrated: true, backendActive: status.is_export_active };
      // Recover a run started before this store existed (reload or another tab).
      if (status.is_export_active && !state.isExporting && !state.ownsRun) {
        return {
          ...base,
          isExporting: true,
          phase: "exporting" as const,
          startedAt: state.startedAt ?? Date.now(),
        };
      }
      if (!status.is_export_active && state.isExporting && !state.ownsRun) {
        // A standalone load_checkpoint must never settle as a finished export.
        const wasExport =
          !!status.last_op_kind && status.last_op_kind !== "load_checkpoint";
        if (status.last_op_status === "error") {
          return {
            ...base,
            isExporting: false,
            phase: "error" as const,
            error: status.last_op_error ?? "Export failed",
          };
        }
        if (status.last_op_status === "cancelled") {
          return { ...base, isExporting: false, phase: "canceled" as const };
        }
        if (status.last_op_status === "success" && wasExport) {
          return {
            ...base,
            isExporting: false,
            phase: "success" as const,
            result: {
              outputPath: status.last_op_output_path ?? null,
              outputPaths: status.last_op_output_path
                ? [{ label: "", path: status.last_op_output_path }]
                : [],
              destination: state.result?.destination ?? "local",
            },
          };
        }
        return { ...base, isExporting: false, phase: "idle" as const };
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
      // The in-flight POST still rejects when the worker dies, giving the canceled phase.
    }
  },

  runExport: async (params) => {
    const runId = get().runId + 1;
    const quantTotal =
      params.exportMethod === "gguf"
        ? Math.max(1, params.quantLevels.length)
        : params.exportMethod === "merged"
          ? Math.max(1, params.mergedSelections?.length ?? 1)
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
      reconnecting: false,
      startedAt: Date.now(),
      result: null,
      error: null,
      cancelRequested: false,
    });

    const isCurrent = () => get().runId === runId;
    const pushToHub = params.destination === "hub";

    // Survive a Cloudflare 524 by settling a recoverable failure through status polls.
    const runRecoverableOp = async (
      post: () => Promise<ExportOperationResponse>,
    ): Promise<{
      outputPath: string | null;
      details?: ExportOperationResponse["details"];
    }> => {
      let baseline: number | null = null;
      try {
        baseline = (await getExportStatus()).last_op_seq ?? 0;
      } catch {
        baseline = null;
      }
      try {
        const resp = await post();
        return {
          outputPath: resp.details?.output_path ?? null,
          details: resp.details,
        };
      } catch (err) {
        if (!isRecoverableTransportError(err)) throw err;
        set({ reconnecting: true });
        try {
          return await recoverViaStatus(baseline, isCurrent);
        } finally {
          if (isCurrent()) set({ reconnecting: false });
        }
      }
    };

    try {
      if (params.sourceMode === "checkpoint") {
        if (!params.checkpointPath) {
          throw new Error("No checkpoint selected");
        }
        const checkpointPath = params.checkpointPath;
        await runRecoverableOp(() =>
          loadCheckpoint({
            checkpoint_path: checkpointPath,
            hf_token: params.loadToken ?? null,
          }),
        );
      } else {
        await runRecoverableOp(() =>
          loadCheckpoint({
            checkpoint_path: params.source,
            load_in_4bit: false,
            trust_remote_code:
              params.modelSource === "hf" ? params.trustRemoteCode : true,
            approved_remote_code_fingerprint:
              params.approvedRemoteCodeFingerprint ?? null,
            hf_token: params.loadToken ?? null,
          }),
        );
      }
      if (!isCurrent()) return;

      set({ phase: "exporting" });
      const outputs: { label: string; path: string }[] = [];

      if (params.exportMethod === "merged") {
        const selections =
          params.mergedSelections && params.mergedSelections.length > 0
            ? params.mergedSelections
            : [{ formatType: "16-bit (FP16)", compressedMethod: null, label: "16-bit" }];
        for (let i = 0; i < selections.length; i += 1) {
          if (!isCurrent()) return;
          set({ quantIndex: i });
          const sel = selections[i];
          const { outputPath } = await runRecoverableOp(() =>
            exportMerged({
              save_directory: params.saveDirectory,
              format_type: sel.formatType,
              compressed_method: sel.compressedMethod,
              push_to_hub: pushToHub,
              repo_id: params.repoId,
              hf_token: params.token,
              private: params.privateRepo,
              install_missing_dependencies: Boolean(
                params.installMissingDependencies,
              ),
            }),
          );
          if (outputPath) outputs.push({ label: sel.label, path: outputPath });
          if (!isCurrent()) return;
          set({ quantIndex: i + 1 });
        }
      } else if (params.exportMethod === "gguf") {
        // Send the whole quant list in ONE call: the model is merged once and every GGUF comes
        // from that single merge (unsloth save_to_gguf loops internally).
        const { outputPath, details } = await runRecoverableOp(() =>
          exportGGUF({
            save_directory: params.saveDirectory,
            quantization_method: params.quantLevels,
            push_to_hub: pushToHub,
            repo_id: params.repoId,
            // Fall back to the load token; both are the same HF token.
            hf_token: params.token ?? params.loadToken ?? null,
            imatrix: params.useImatrix,
            imatrix_path: params.useImatrix
              ? params.imatrixPath?.trim() || null
              : null,
            private: params.privateRepo,
            npu_q4nx: params.npuQ4nx,
          }),
        );
        if (outputPath) {
          outputs.push({
            label: params.decisionOutputLabel
              ? params.decisionOutputLabel(
                  decisionQuantizations(details, params.quantLevels),
                )
              : "GGUF",
            path: outputPath,
          });
        }
        if (outputPath && params.npuQ4nx) {
          const sep = outputPath.includes("\\") ? "\\" : "/";
          outputs.push({ label: "AMD NPU (Q4NX)", path: `${outputPath}${sep}npu-q4nx` });
        }
        if (!isCurrent()) return;
        set({ quantIndex: get().quantTotal });
      } else if (params.exportMethod === "lora") {
        const { outputPath } = await runRecoverableOp(() =>
          exportLoRA({
            save_directory: params.saveDirectory,
            push_to_hub: pushToHub,
            repo_id: params.repoId,
            // Fall back to the load token; both are the same HF token.
            hf_token: params.token ?? params.loadToken ?? null,
            private: params.privateRepo,
            gguf: params.loraGguf ?? false,
            gguf_outtype: params.loraGgufOuttype ?? "q8_0",
            adapter_format: params.adapterFormat,
          }),
        );
        if (outputPath) {
          outputs.push({
            label: params.loraGguf ? "GGUF LoRA adapter" : "LoRA adapter",
            path: outputPath,
          });
        }
      }
      if (!isCurrent()) return;

      set({
        phase: "success",
        isExporting: false,
        reconnecting: false,
        result: {
          outputPath: outputs[0]?.path ?? null,
          outputPaths: outputs,
          destination: params.destination,
        },
      });
    } catch (err) {
      if (!isCurrent()) return;
      if (get().cancelRequested || err instanceof ExportCanceledError) {
        set({
          phase: "canceled",
          isExporting: false,
          reconnecting: false,
          error: null,
        });
      } else {
        set({
          phase: "error",
          isExporting: false,
          reconnecting: false,
          error: err instanceof Error ? err.message : "Export failed",
        });
      }
    } finally {
      // Only the run that still owns the store releases ownership and frees the worker.
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

/** No byte-level signal exists, so progress is phase plus completed-quant based. */
export function selectExportProgressPercent(state: ExportRuntimeStore): number {
  const total = Math.max(1, state.quantTotal);
  const exportBand = () => {
    const done = Math.min(Math.max(state.quantIndex, 0), total);
    return Math.round(15 + (done / total) * 72);
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

export function isExportPanelActive(state: ExportRuntimeStore): boolean {
  return state.isExporting || state.phase !== "idle";
}
