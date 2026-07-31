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

/** Thrown by status recovery when the backend reports the op was cancelled. */
class ExportCanceledError extends Error {
  constructor() {
    super("Export canceled");
    this.name = "ExportCanceledError";
  }
}

// Status-recovery tuning (used when a blocking export POST is cut off by a
// Cloudflare tunnel 524 while the backend op keeps running).
const RECOVERY_POLL_INTERVAL_MS = 1500;
const RECOVERY_GRACE_MS = 15000; // wait this long for the op to appear on status
const RECOVERY_MAX_MS = 2 * 60 * 60 * 1000; // hard cap (exports can be long)
const RECOVERY_MAX_STATUS_FAILS = 5; // give up if status itself is unreachable

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Settle an export phase whose blocking POST was cut off by a tunnel timeout, by
 * polling /api/export/status until the still-running backend op finishes, then
 * reading its recorded outcome. `baseline` is `last_op_seq` captured before the
 * POST; a record is "ours" once the op went inactive AND either we observed it
 * active or its seq advanced past the baseline (single serial driver ⇒ exact).
 */
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
      // Status itself is transiently unreachable; tolerate a few, then (past the
      // grace window) give up so a truly-down backend surfaces a real error.
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

    // Inactive but our op was never observed: the POST likely died before the
    // backend started it. Allow a grace window for the op to appear, then fail.
    if (Date.now() - start > RECOVERY_GRACE_MS) {
      throw new Error(
        "The export request failed before the server started the operation.",
      );
    }
  }
  throw new Error("Timed out waiting for the export to finish.");
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

/** Snapshot of what is being exported, captured when a run starts. */
export interface ExportRunSummary {
  baseModelName: string;
  checkpointLabel: string | null;
  methodLabel: string;
  method: ExportMethod;
  quantLevels: string[];
  /** Merged: the selected format values (for the summary "Formats" row and to reseed the picker). */
  mergedFormats: string[];
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
  /** Consent fingerprint from the load-time remote-code review dialog (HF custom code). */
  approvedRemoteCodeFingerprint?: string | null;
  /** HF token for loading a gated/private source model (separate from the Hub upload token). */
  loadToken?: string | null;
  exportMethod: ExportMethod;
  isAdapter: boolean;
  quantLevels: string[];
  /** GGUF: use an importance matrix (auto-download); required for the IQ quants. */
  useImatrix?: boolean;
  /** Merged: precision formats, each exported to its own sibling directory. Defaults to 16-bit.
   *  `label` is the display name for the success banner's per-format output line. */
  mergedSelections?: {
    formatType: string;
    compressedMethod: string | null;
    label: string;
  }[];
  /** LoRA: also emit a GGUF LoRA adapter (llama.cpp `--lora`), and its output float type. */
  loraGguf?: boolean;
  loraGgufOuttype?: string;
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
  /** True while a phase POST's response was lost (tunnel timeout) and we are
   *  settling the run by polling /api/export/status instead. Logs keep streaming. */
  reconnecting: boolean;
  startedAt: number | null;
  /** `outputPath` is the first path (back-compat); `outputPaths` is one entry per written folder
   *  so a multi-format merged run can list every sibling directory it created. */
  result: {
    outputPath: string | null;
    outputPaths: { label: string; path: string }[];
    destination: ExportDestination;
  } | null;
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
  /** Append one streamed line (SSE). De-duped by seq against the poll path. */
  appendLog: (entry: ExportLogEntry, seq?: number) => void;
  /** Merge a batch of polled lines (JSON fallback). De-duped by seq. */
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
      // De-dupe by seq: the SSE stream and the JSON poll fallback both feed
      // logs, so ignore anything at or below the highest seq already seen.
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
        // `status` lines are the worker's high-level progress markers; surface
        // the most recent one as the stage label.
        stage: entry.stream === "status" ? entry.line : state.stage,
      };
    }),

  appendLogs: (entries) =>
    set((state) => {
      // Keep only lines newer than the highest seq we've shown (covers overlap
      // with the SSE stream and with the previous poll batch).
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

      // Latest `status` line in the batch becomes the stage label.
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
      // A recovered (not store-owned) run finished on the backend. Settle from
      // the last-op record when present (accurate success/error/output path),
      // else fall back to the optimistic guess.
      if (!status.is_export_active && state.isExporting && !state.ownsRun) {
        // A standalone load_checkpoint (or no recorded op) is not an export and
        // must never settle as a finished export. A completed export ends on its
        // export_* op or the trailing cleanup, both of which count.
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
              // A run recovered from the backend only knows the last output path.
              outputPaths: status.last_op_output_path
                ? [{ label: "", path: status.last_op_output_path }]
                : [],
              destination: state.result?.destination ?? "local",
            },
          };
        }
        // Load-only op, or no clear success record: nothing was exported.
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
      // Best-effort: the in-flight export POST will still reject when the
      // worker dies, which runExport turns into the canceled phase.
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

    // Run a phase POST so it survives a Cloudflare tunnel 524: capture the
    // last-op baseline, fire the POST, and on a recoverable transport failure
    // settle the still-running backend op via short status polls instead of
    // failing. Returns the resolved output path (null for load/hub-only).
    const runRecoverableOp = async (
      post: () => Promise<ExportOperationResponse>,
    ): Promise<{ outputPath: string | null }> => {
      let baseline: number | null = null;
      try {
        baseline = (await getExportStatus()).last_op_seq ?? 0;
      } catch {
        baseline = null; // pre-read failed; recovery falls back to "saw active"
      }
      try {
        const resp = await post();
        return { outputPath: resp.details?.output_path ?? null };
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
      // 1. Load the model source into a fresh export subprocess.
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

      // 2. Run the export. Collect every resolved output_path so the success
      // banner can list each sibling directory a multi-format run created.
      set({ phase: "exporting" });
      const outputs: { label: string; path: string }[] = [];

      if (params.exportMethod === "merged") {
        // Each selected format writes its own sibling directory (PEFT or non-PEFT base alike).
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
            }),
          );
          if (outputPath) outputs.push({ label: sel.label, path: outputPath });
          if (!isCurrent()) return;
          set({ quantIndex: i + 1 });
        }
      } else if (params.exportMethod === "gguf") {
        // Send the whole quant list in ONE call: the model is merged once and every GGUF comes
        // from that single merge (unsloth save_to_gguf loops internally).
        const { outputPath } = await runRecoverableOp(() =>
          exportGGUF({
            save_directory: params.saveDirectory,
            quantization_method: params.quantLevels,
            push_to_hub: pushToHub,
            repo_id: params.repoId,
            hf_token: params.token,
            imatrix: params.useImatrix,
          }),
        );
        if (outputPath) outputs.push({ label: "GGUF", path: outputPath });
        if (!isCurrent()) return;
        set({ quantIndex: get().quantTotal });
      } else if (params.exportMethod === "lora") {
        const { outputPath } = await runRecoverableOp(() =>
          exportLoRA({
            save_directory: params.saveDirectory,
            push_to_hub: pushToHub,
            repo_id: params.repoId,
            // A local GGUF LoRA export still reloads a possibly-gated base config, so fall back to
            // the load token when there is no hub-upload token (both are the same HF token).
            hf_token: params.token ?? params.loadToken ?? null,
            private: params.privateRepo,
            gguf: params.loraGguf ?? false,
            gguf_outtype: params.loraGgufOuttype ?? "q8_0",
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
