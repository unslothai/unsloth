// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The run lives on the server; this keeps the tab's view of it and owns the model swap
// around it, so leaving the tab mid-run still puts chat's model back afterwards.

import {
  type InferenceStatusResponse,
  getInferenceStatus,
  loadModel,
  useChatRuntimeStore,
} from "@/features/chat";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { restore } from "../api/bench-runner";
import { releaseBenchLease } from "../api/bench-runs-api";
import { chatBaseLoad } from "../api/chat-base";
import {
  type LlamaBenchConfig,
  type LlamaBenchJob,
  type SavedLlamaBenchRun,
  cancelLlamaBench,
  deleteLlamaBenchRun,
  getLlamaBenchJob,
  getLlamaBenchStatus,
  listLlamaBenchRuns,
  startLlamaBench,
} from "./llama-bench-api";
import { ownsRestore, shouldPutBack } from "./llama-bench-restore";

const DEFAULT_LLAMA_BENCH: LlamaBenchConfig = {
  prompt_tokens: [512],
  gen_tokens: [128],
  depths: [0],
  repetitions: 5,
  flash_attn: "auto",
};

const TERMINAL = new Set(["done", "error", "cancelled"]);
const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

interface LlamaBenchState {
  config: LlamaBenchConfig;
  available: boolean | null;
  job: LlamaBenchJob | null;
  /** Client-side phase around the server job: swapping the model in, or putting it back. */
  phase: "idle" | "loading" | "running" | "restoring";
  error: string | null;
  runs: SavedLlamaBenchRun[];
  shownId: string | null;
  /** Chat's model before the swap, persisted so a reload mid-run can still put it back. */
  pendingRestore: PendingRestore | null;
  setConfig: (patch: Partial<LlamaBenchConfig>) => void;
  refresh: () => Promise<void>;
  start: (model: string | null, variant: string | null) => Promise<void>;
  cancel: () => Promise<void>;
  show: (id: string | null) => void;
  remove: (id: string) => Promise<void>;
}

interface PendingRestore {
  original: InferenceStatusResponse;
  originalLoad: ReturnType<typeof chatBaseLoad>;
  /** The run it belongs to; null until the server has answered with one. */
  jobId: string | null;
  benched: { model: string | null; variant: string | null };
}

const POLL_MISS_LIMIT = 30;

async function putChatBack(
  set: (s: Partial<LlamaBenchState>) => void,
  get: () => LlamaBenchState,
): Promise<void> {
  const pending = get().pendingRestore;
  // The server keeps the run's lease (the job id) until this load, so no other run gets in first.
  const holder = get().job?.id;
  if (!pending) {
    if (holder) await releaseBenchLease(holder);
    return;
  }
  set({ phase: "restoring", pendingRestore: null });
  try {
    const now = await getInferenceStatus();
    // Another tab's restore, or a model the user loaded since, stays.
    if (
      shouldPutBack(
        { model: now.active_model, variant: now.gguf_variant },
        {
          model: pending.original.active_model,
          variant: pending.original.gguf_variant,
        },
        pending.benched,
      )
    )
      await restore(pending.original, pending.originalLoad, holder);
  } catch (err) {
    set({
      error: `The run finished, but chat's model didn't load back: ${
        err instanceof Error ? err.message : String(err)
      }`,
    });
  } finally {
    if (holder) await releaseBenchLease(holder);
  }
}

async function pollUntilDone(
  set: (s: Partial<LlamaBenchState>) => void,
): Promise<LlamaBenchJob | null> {
  let misses = 0;
  for (;;) {
    let job: LlamaBenchJob | null;
    try {
      job = await getLlamaBenchJob();
      misses = 0;
    } catch (err) {
      // A failed read is not a finished job: returning would restore chat over a running bench.
      if (++misses >= POLL_MISS_LIMIT) throw err;
      await sleep(1000);
      continue;
    }
    if (job) set({ job });
    if (!job || TERMINAL.has(job.status)) return job;
    await sleep(1000);
  }
}

export const useLlamaBenchStore = create<LlamaBenchState>()(
  persist(
    (set, get) => ({
      config: DEFAULT_LLAMA_BENCH,
      available: null,
      job: null,
      phase: "idle",
      error: null,
      runs: [],
      shownId: null,
      pendingRestore: null,
      setConfig: (patch) => set({ config: { ...get().config, ...patch } }),
      refresh: async () => {
        const [status, runs] = await Promise.all([
          getLlamaBenchStatus().catch(() => null),
          listLlamaBenchRuns().catch(() => get().runs),
        ]);
        set({ available: status?.available ?? null, runs });
        if (!status || get().phase !== "idle") return;
        // A run started in another tab or before a reload: follow it, then put chat's model back.
        if (status.job) set({ job: status.job });
        const running = status.job?.status === "running";
        const owned = ownsRestore(get().pendingRestore?.jobId, status.job?.id);
        if (!owned) set({ pendingRestore: null });
        if (!running && !owned) return;
        set({ phase: "running" });
        try {
          if (running) await pollUntilDone(set);
          await putChatBack(set, get);
        } finally {
          set({ phase: "idle", runs: await listLlamaBenchRuns().catch(() => get().runs) });
        }
      },
      start: async (model, variant) => {
        if (get().phase !== "idle") return;
        set({ error: null, shownId: null, job: null, phase: "loading" });
        let status: Awaited<ReturnType<typeof getInferenceStatus>>;
        try {
          await useChatRuntimeStore.getState().hydratePersistedSettings();
          status = await getInferenceStatus();
        } catch (err) {
          // Nothing touched yet, so straight back to idle.
          set({ phase: "idle", error: err instanceof Error ? err.message : String(err) });
          return;
        }
        const original = status;
        const originalLoad = chatBaseLoad(original);
        const pending = (jobId: string | null): PendingRestore => ({
          original,
          originalLoad,
          jobId,
          benched: {
            model: model ?? original.active_model,
            variant: model ? variant : (original.gguf_variant ?? null),
          },
        });
        let touched = false;
        try {
          if (
            model &&
            (model !== status.active_model ||
              (variant ?? null) !== (status.gguf_variant ?? null))
          ) {
            touched = true;
            if (original.active_model) set({ pendingRestore: pending(null) });
            await loadModel(
              {
                ...chatBaseLoad({
                  ...status,
                  active_model: model,
                  gguf_variant: variant,
                }),
                gguf_variant: variant,
                force_reload: true,
              },
              { runtime: "chat" },
            );
            status = await getInferenceStatus();
          }
          if (!status.active_model || status.is_gguf === false)
            throw new Error(
              "Load a GGUF model first. llama-bench measures the model chat has loaded.",
            );
          // Before the POST: the server unloads chat's model, and a lost answer must still restore it.
          touched = true;
          if (original.active_model) set({ pendingRestore: pending(null) });
          const job = await startLlamaBench(get().config);
          if (original.active_model) set({ pendingRestore: pending(job.id) });
          set({ job, phase: "running" });
          const done = await pollUntilDone(set);
          if (done?.status === "error")
            set({ error: done.error ?? "llama-bench failed" });
        } catch (err) {
          set({ error: err instanceof Error ? err.message : String(err) });
        } finally {
          if (touched && original.active_model) {
            set({ pendingRestore: pending(get().job?.id ?? null) });
            await putChatBack(set, get);
          } else {
            const held = get().job?.id;
            if (held) await releaseBenchLease(held);
          }
          set({
            phase: "idle",
            runs: await listLlamaBenchRuns().catch(() => get().runs),
          });
        }
      },
      cancel: async () => {
        await cancelLlamaBench().catch(() => undefined);
      },
      show: (id) => set({ shownId: id }),
      remove: async (id) => {
        await deleteLlamaBenchRun(id);
        set({
          runs: get().runs.filter((r) => r.id !== id),
          shownId: get().shownId === id ? null : get().shownId,
        });
      },
    }),
    {
      name: "unsloth_llama_bench",
      partialize: (s) => ({ config: s.config, pendingRestore: s.pendingRestore }),
    },
  ),
);
