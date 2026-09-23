// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sweep being set up, the one running, and past runs. The runner lives here rather
// than in the page, so a sweep keeps going while the user is in chat.

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { BenchSetupError, runBenchmark } from "../api/bench-runner";
import {
  type BenchConfig,
  type BenchRun,
  type SweepKind,
  type Variant,
  defaultBaseline,
  sweepVariants,
} from "../lib/bench-math";

const MAX_HISTORY = 20;

export function presetConfig(sweep: SweepKind, maxContext?: number | null): Pick<BenchConfig, "sweep" | "variants" | "baseline"> {
  const variants = sweepVariants(sweep, maxContext);
  return { sweep, variants, baseline: defaultBaseline(variants) };
}

export const DEFAULT_CONFIG: BenchConfig = {
  ...presetConfig("spec"),
  promptSet: "chat",
  customPrompt: "",
  rotatePrompts: true,
  maxTokens: 256,
  warmup: 1,
  repetitions: 3,
  temperature: 0.7,
  seed: 3407,
  restoreAfter: true,
};

interface LiveRun {
  run: BenchRun;
  progress: string;
}

interface BenchmarksState {
  config: BenchConfig;
  /** Variant labels switched off for the next run. */
  disabled: string[];
  history: BenchRun[];
  selectedRunId: string | null;
  live: LiveRun | null;
  error: string | null;
  setConfig: (patch: Partial<BenchConfig>) => void;
  choosePreset: (sweep: SweepKind, maxContext?: number | null) => void;
  toggleVariant: (label: string) => void;
  selectRun: (id: string | null) => void;
  deleteRun: (id: string) => void;
  start: () => Promise<void>;
  cancel: () => void;
}

let controller: AbortController | null = null;

export const useBenchmarksStore = create<BenchmarksState>()(
  persist(
    (set, get) => ({
      config: DEFAULT_CONFIG,
      disabled: [],
      history: [],
      selectedRunId: null,
      live: null,
      error: null,
      setConfig: (patch) => set((s) => ({ config: { ...s.config, ...patch } })),
      choosePreset: (sweep, maxContext) => set((s) => ({ config: { ...s.config, ...presetConfig(sweep, maxContext) }, disabled: [] })),
      toggleVariant: (label) =>
        set((s) => ({ disabled: s.disabled.includes(label) ? s.disabled.filter((x) => x !== label) : [...s.disabled, label] })),
      selectRun: (id) => set({ selectedRunId: id }),
      deleteRun: (id) =>
        set((s) => ({
          history: s.history.filter((r) => r.id !== id),
          selectedRunId: s.selectedRunId === id ? null : s.selectedRunId,
        })),
      start: async () => {
        if (get().live) return;
        const { config, disabled } = get();
        const variants: Variant[] = config.variants.filter((v) => !disabled.includes(v.label));
        if (variants.length === 0) {
          set({ error: "Switch on at least one setting to run." });
          return;
        }
        const effective: BenchConfig = {
          ...config,
          variants,
          baseline: config.baseline && variants.some((v) => v.label === config.baseline) ? config.baseline : null,
        };
        controller = new AbortController();
        const placeholder: BenchRun = {
          id: "pending",
          createdAt: Date.now(),
          finishedAt: null,
          model: "",
          ggufVariant: null,
          kv: null,
          context: null,
          config: effective,
          meta: {},
          outcomes: variants.map((v) => ({ label: v.label, state: "queued" })),
          results: [],
        };
        set({ live: { run: placeholder, progress: "Reading the loaded model" }, error: null, selectedRunId: null });
        const patchLive = (fn: (run: BenchRun) => BenchRun, progress?: string) =>
          set((s) => (s.live ? { live: { run: fn(s.live.run), progress: progress ?? s.live.progress } } : s));
        try {
          const run = await runBenchmark(
            effective,
            {
              onOutcome: (o) => patchLive((r) => ({ ...r, outcomes: r.outcomes.map((x) => (x.label === o.label ? o : x)) })),
              onResult: (res) => patchLive((r) => ({ ...r, results: [...r.results, res] })),
              onProgress: (text) => patchLive((r) => r, text),
            },
            controller.signal,
          );
          set((s) => ({
            live: null,
            history: run.results.length ? [run, ...s.history].slice(0, MAX_HISTORY) : s.history,
            selectedRunId: run.results.length ? run.id : s.selectedRunId,
          }));
        } catch (err) {
          set({
            live: null,
            error: err instanceof BenchSetupError || err instanceof Error ? err.message : String(err),
          });
        } finally {
          controller = null;
        }
      },
      cancel: () => {
        controller?.abort();
        set((s) => (s.live ? { live: { ...s.live, progress: "Stopping after the current step" } } : s));
      },
    }),
    {
      name: "unsloth_benchmarks",
      version: 1,
      partialize: (s) => ({ config: s.config, disabled: s.disabled, history: s.history }),
      merge: (persisted, current) => {
        const saved = (persisted ?? {}) as Partial<BenchmarksState>;
        return {
          ...current,
          config: { ...DEFAULT_CONFIG, ...(saved.config ?? {}) },
          disabled: Array.isArray(saved.disabled) ? saved.disabled : [],
          history: Array.isArray(saved.history) ? saved.history : [],
        };
      },
    },
  ),
);
