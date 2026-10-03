// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The run lives on the server; this keeps the tab's view of it and owns the model swap
// around it, so leaving the tab mid-run still puts chat's model back afterwards.

import {
  getInferenceStatus,
  loadModel,
  useChatRuntimeStore,
} from "@/features/chat";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { restore } from "../api/bench-runner";
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

export const DEFAULT_LLAMA_BENCH: LlamaBenchConfig = {
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
  setConfig: (patch: Partial<LlamaBenchConfig>) => void;
  refresh: () => Promise<void>;
  start: (model: string | null, variant: string | null) => Promise<void>;
  cancel: () => Promise<void>;
  show: (id: string | null) => void;
  remove: (id: string) => Promise<void>;
}

async function pollUntilDone(
  set: (s: Partial<LlamaBenchState>) => void,
): Promise<LlamaBenchJob | null> {
  for (;;) {
    const job = await getLlamaBenchJob().catch(() => null);
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
      setConfig: (patch) => set({ config: { ...get().config, ...patch } }),
      refresh: async () => {
        const [status, runs] = await Promise.all([
          getLlamaBenchStatus().catch(() => null),
          listLlamaBenchRuns().catch(() => get().runs),
        ]);
        set({ available: status?.available ?? null, runs });
        // A run started in another tab or before a reload: follow it.
        if (status?.job && get().phase === "idle") {
          set({ job: status.job });
          if (status.job.status === "running") {
            set({ phase: "running" });
            await pollUntilDone(set);
            set({ phase: "idle", runs: await listLlamaBenchRuns() });
          }
        }
      },
      start: async (model, variant) => {
        if (get().phase !== "idle") return;
        set({ error: null, shownId: null, phase: "loading" });
        await useChatRuntimeStore.getState().hydratePersistedSettings();
        let status = await getInferenceStatus();
        const original = status;
        const originalLoad = chatBaseLoad(original);
        let touched = false;
        try {
          if (
            model &&
            (model !== status.active_model ||
              (variant ?? null) !== (status.gguf_variant ?? null))
          ) {
            touched = true;
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
          const job = await startLlamaBench(get().config);
          // The server unloaded chat's model to give llama-bench the GPU.
          touched = true;
          set({ job, phase: "running" });
          const done = await pollUntilDone(set);
          if (done?.status === "error")
            set({ error: done.error ?? "llama-bench failed" });
        } catch (err) {
          set({ error: err instanceof Error ? err.message : String(err) });
        } finally {
          if (touched && original.active_model) {
            set({ phase: "restoring" });
            await restore(original, originalLoad).catch((err) =>
              set({
                error: `The run finished, but chat's model didn't load back: ${
                  err instanceof Error ? err.message : String(err)
                }`,
              }),
            );
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
      name: "unsloth-llama-bench",
      partialize: (s) => ({ config: s.config }),
    },
  ),
);
