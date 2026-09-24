// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sweep being set up, the one running, and the runs on record. The runner lives here
// rather than in the page, so a sweep keeps going while the user is in chat. Runs are
// saved to studio.db as they go; only the setup sheet stays in localStorage.

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { applyVariantToChat } from "../api/apply-to-chat";
import { BenchSetupError, runBenchmark } from "../api/bench-runner";
import {
  type BenchRunSummary,
  deleteBenchRun,
  getBenchRun,
  listBenchRuns,
  saveBenchRun,
} from "../api/bench-runs-api";
import {
  type BenchConfig,
  type BenchRun,
  type SweepKind,
  type Variant,
  aggregate,
  defaultBaseline,
  sweepVariants,
  tuneVerdict,
} from "../lib/bench-math";

export type BenchKind = "sweep" | "tune";

export function presetConfig(
  sweep: SweepKind,
  maxContext?: number | null,
): Pick<BenchConfig, "sweep" | "variants" | "baseline"> {
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
  /** Newest first, from studio.db. */
  runs: BenchRunSummary[];
  runsLoaded: boolean;
  /** Full runs already fetched, by id. */
  loaded: Record<string, BenchRun>;
  selectedRunId: string | null;
  live: LiveRun | null;
  error: string | null;
  /** The run whose winner is being handed to chat, while the reload is in flight. */
  applying: string | null;
  setConfig: (patch: Partial<BenchConfig>) => void;
  chooseKind: (kind: BenchKind) => void;
  choosePreset: (sweep: SweepKind, maxContext?: number | null) => void;
  applyToChat: (run: BenchRun) => Promise<void>;
  toggleVariant: (label: string) => void;
  refreshRuns: () => Promise<void>;
  selectRun: (id: string | null) => Promise<void>;
  deleteRun: (id: string) => Promise<void>;
  start: () => Promise<void>;
  cancel: () => void;
}

let controller: AbortController | null = null;

/** Saves collapse to one in flight plus one pending, so a fast sweep never queues a save per token. */
function makeSaver(): (run: BenchRun) => Promise<void> {
  let inFlight: Promise<void> | null = null;
  let pending: BenchRun | null = null;
  const flush = async (): Promise<void> => {
    while (pending) {
      const next = pending;
      pending = null;
      try {
        await saveBenchRun(next);
      } catch {
        // The final save reports; a missed mid-run save is caught by the next one.
      }
    }
  };
  return (run) => {
    pending = run;
    if (!inFlight) inFlight = flush().finally(() => (inFlight = null));
    return inFlight;
  };
}

export const useBenchmarksStore = create<BenchmarksState>()(
  persist(
    (set, get) => ({
      config: DEFAULT_CONFIG,
      disabled: [],
      runs: [],
      runsLoaded: false,
      loaded: {},
      selectedRunId: null,
      live: null,
      error: null,
      applying: null,
      setConfig: (patch) => set((s) => ({ config: { ...s.config, ...patch } })),
      chooseKind: (kind) => {
        const { config, choosePreset } = get();
        if (kind === "tune" && config.sweep !== "tune") choosePreset("tune");
        if (kind === "sweep" && config.sweep === "tune") choosePreset("spec");
      },
      choosePreset: (sweep, maxContext) =>
        set((s) => ({
          config: { ...s.config, ...presetConfig(sweep, maxContext) },
          disabled: [],
        })),
      applyToChat: async (run) => {
        const verdict = tuneVerdict(
          aggregate(run.results, run.config.variants, null),
          run.config.variants,
        );
        const variant =
          verdict &&
          run.config.variants.find((v) => v.label === verdict.pick.label);
        if (!variant) return;
        set({ applying: run.id, error: null });
        try {
          await applyVariantToChat(variant, run.config.tuneModel ?? run.model);
        } catch (err) {
          set({
            error: `Could not apply to chat: ${err instanceof Error ? err.message : String(err)}`,
          });
        } finally {
          set({ applying: null });
        }
      },
      toggleVariant: (label) =>
        set((s) => ({
          disabled: s.disabled.includes(label)
            ? s.disabled.filter((x) => x !== label)
            : [...s.disabled, label],
        })),
      refreshRuns: async () => {
        try {
          const runs = await listBenchRuns();
          set({ runs, runsLoaded: true });
        } catch (err) {
          set({
            runsLoaded: true,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      },
      selectRun: async (id) => {
        set({ selectedRunId: id });
        if (!id || get().loaded[id]) return;
        try {
          const run = await getBenchRun(id);
          set((s) => ({ loaded: { ...s.loaded, [id]: run } }));
        } catch (err) {
          set({ error: err instanceof Error ? err.message : String(err) });
        }
      },
      deleteRun: async (id) => {
        set((s) => {
          const loaded = { ...s.loaded };
          delete loaded[id];
          return {
            runs: s.runs.filter((r) => r.id !== id),
            loaded,
            selectedRunId: s.selectedRunId === id ? null : s.selectedRunId,
          };
        });
        try {
          await deleteBenchRun(id);
        } catch (err) {
          set({ error: err instanceof Error ? err.message : String(err) });
        }
      },
      start: async () => {
        if (get().live) return;
        const { config, disabled } = get();
        const variants: Variant[] = config.variants.filter(
          (v) => !disabled.includes(v.label),
        );
        if (variants.length === 0) {
          set({ error: "Switch on at least one setting to run." });
          return;
        }
        const effective: BenchConfig = {
          ...config,
          variants,
          baseline:
            config.baseline && variants.some((v) => v.label === config.baseline)
              ? config.baseline
              : null,
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
        set({
          live: { run: placeholder, progress: "Reading the loaded model" },
          error: null,
        });
        const save = makeSaver();
        const patchLive = (
          fn: (run: BenchRun) => BenchRun,
          progress?: string,
        ) =>
          set((s) => {
            if (!s.live) return s;
            const run = fn(s.live.run);
            // Saved as it goes, so a crash mid-sweep keeps the rows that finished.
            if (run.id !== "pending" && run.results.length) void save(run);
            return { live: { run, progress: progress ?? s.live.progress } };
          });
        try {
          const run = await runBenchmark(
            effective,
            {
              onStart: (run) => patchLive(() => run),
              onOutcome: (o) =>
                patchLive((r) => ({
                  ...r,
                  outcomes: r.outcomes.map((x) =>
                    x.label === o.label ? o : x,
                  ),
                })),
              onResult: (res) =>
                patchLive((r) => ({ ...r, results: [...r.results, res] })),
              onProgress: (text) => patchLive((r) => r, text),
            },
            controller.signal,
          );
          if (run.results.length) {
            let saved = run;
            try {
              saved = await saveBenchRun(run);
            } catch (err) {
              set({
                error: `The run finished but could not be saved: ${err instanceof Error ? err.message : String(err)}`,
              });
            }
            set((s) => ({
              live: null,
              loaded: { ...s.loaded, [saved.id]: saved },
              selectedRunId: saved.id,
            }));
            await get().refreshRuns();
          } else {
            set({ live: null });
          }
        } catch (err) {
          set({
            live: null,
            error:
              err instanceof BenchSetupError || err instanceof Error
                ? err.message
                : String(err),
          });
        } finally {
          controller = null;
        }
      },
      cancel: () => {
        controller?.abort();
        set((s) =>
          s.live
            ? {
                live: {
                  ...s.live,
                  progress: "Stopping after the current step",
                },
              }
            : s,
        );
      },
    }),
    {
      name: "unsloth_benchmarks",
      version: 2,
      partialize: (s) => ({ config: s.config, disabled: s.disabled }),
      merge: (persisted, current) => {
        const saved = (persisted ?? {}) as Partial<BenchmarksState>;
        return {
          ...current,
          config: { ...DEFAULT_CONFIG, ...(saved.config ?? {}) },
          disabled: Array.isArray(saved.disabled) ? saved.disabled : [],
        };
      },
    },
  ),
);
