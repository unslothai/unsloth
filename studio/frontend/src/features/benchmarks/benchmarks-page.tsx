// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Benchmarks: sweep load settings on the loaded GGUF and chart what each one does to
// throughput, draft acceptance and load time on this machine. One page: the setup card,
// then the run, live or picked from history.

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  type InferenceStatusResponse,
  getInferenceStatus,
} from "@/features/chat";
import { authFetch } from "@/features/auth";
import { cn } from "@/lib/utils";
import {
  Clock01Icon,
  CpuIcon,
  DashboardSpeed01Icon,
  Delete02Icon,
  Rocket01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import { RunResults } from "./components/results-panel";
import { ModelStrip, SetupPanel, StatPill } from "./components/setup-panel";
import { TuneVerdictCard } from "./components/tune-section";
import { SWEEP_TITLE, modelShort } from "./lib/bench-math";
import { useBenchmarksStore } from "./stores/benchmarks-store";

function useLoadedModel(paused: boolean): InferenceStatusResponse | null {
  const [status, setStatus] = useState<InferenceStatusResponse | null>(null);
  useEffect(() => {
    if (paused) return;
    const controller = new AbortController();
    const read = () =>
      getInferenceStatus(controller.signal)
        .then(setStatus)
        .catch(() => undefined);
    void read();
    const timer = window.setInterval(read, 5000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [paused]);
  return status;
}

interface Machine {
  gpu: string | null;
  vramGb: number | null;
  backend: string | null;
  llamaTag: string | null;
}

const BACKEND_NAME: Record<string, string> = {
  cuda: "CUDA",
  rocm: "ROCm",
  vulkan: "Vulkan",
  metal: "Metal",
  cpu: "CPU",
  sycl: "SYCL",
};

/** The Hub's toolbar facts: what this machine benchmarks on. */
function useMachine(): Machine {
  const [m, setM] = useState<Machine>({
    gpu: null,
    vramGb: null,
    backend: null,
    llamaTag: null,
  });
  useEffect(() => {
    const read = async (path: string) => {
      try {
        const res = await authFetch(path);
        return res.ok ? ((await res.json()) as Record<string, unknown>) : null;
      } catch {
        return null;
      }
    };
    void Promise.all([
      read("/api/system/hardware"),
      read("/api/llama/backend"),
    ]).then(([hw, llama]) => {
      const gpu = (hw?.gpu ?? {}) as Record<string, unknown>;
      const backend = typeof llama?.backend === "string" ? llama.backend : null;
      setM({
        gpu: typeof gpu.gpu_name === "string" ? gpu.gpu_name : null,
        vramGb:
          typeof gpu.vram_total_gb === "number" ? gpu.vram_total_gb : null,
        backend: backend
          ? (BACKEND_NAME[backend.toLowerCase()] ?? backend)
          : null,
        llamaTag:
          typeof llama?.installed_tag === "string" ? llama.installed_tag : null,
      });
    });
  }, []);
  return m;
}

function History({ shownId }: { shownId: string | null }): ReactElement | null {
  const runs = useBenchmarksStore((s) => s.runs);
  const live = useBenchmarksStore((s) => s.live);
  const selectRun = useBenchmarksStore((s) => s.selectRun);
  const deleteRun = useBenchmarksStore((s) => s.deleteRun);
  if (runs.length === 0) return null;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <Button
          variant="ghost"
          className="h-10 gap-2 rounded-full px-4"
          disabled={Boolean(live)}
        >
          <HugeiconsIcon
            icon={Clock01Icon}
            strokeWidth={1.75}
            className="size-4"
          />
          History
          <span className="text-muted-foreground tabular-nums">
            {runs.length}
          </span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="max-h-[min(60vh,var(--radix-dropdown-menu-content-available-height))] w-80 overflow-y-auto"
      >
        <DropdownMenuLabel className="text-ui-11 text-muted-foreground">
          Saved runs
        </DropdownMenuLabel>
        {runs.map((run) => (
          <DropdownMenuItem
            key={run.id}
            onSelect={() => void selectRun(run.id)}
            className={cn(
              "group flex items-start gap-2 py-2",
              run.id === shownId && "bg-accent",
            )}
          >
            <span className="flex min-w-0 flex-1 flex-col">
              <span className="truncate text-ui-12p5 font-medium">
                {SWEEP_TITLE[run.config.sweep]}
              </span>
              <span className="truncate text-ui-11p5 text-muted-foreground">
                {modelShort(run.model)} ·{" "}
                {new Date(run.createdAt).toLocaleString(undefined, {
                  month: "short",
                  day: "numeric",
                  hour: "numeric",
                  minute: "2-digit",
                })}
              </span>
            </span>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                void deleteRun(run.id);
              }}
              className="rounded-md p-1 text-muted-foreground opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100"
              aria-label="Delete this run"
            >
              <HugeiconsIcon
                icon={Delete02Icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
            </button>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function BenchmarksPage(): ReactElement {
  const live = useBenchmarksStore((s) => s.live);
  const runs = useBenchmarksStore((s) => s.runs);
  const runsLoaded = useBenchmarksStore((s) => s.runsLoaded);
  const loaded = useBenchmarksStore((s) => s.loaded);
  const selectedRunId = useBenchmarksStore((s) => s.selectedRunId);
  const refreshRuns = useBenchmarksStore((s) => s.refreshRuns);
  const selectRun = useBenchmarksStore((s) => s.selectRun);
  const start = useBenchmarksStore((s) => s.start);
  const cancel = useBenchmarksStore((s) => s.cancel);
  const error = useBenchmarksStore((s) => s.error);
  const config = useBenchmarksStore((s) => s.config);
  const disabled = useBenchmarksStore((s) => s.disabled);
  const status = useLoadedModel(Boolean(live));
  const machine = useMachine();
  const maxContext =
    status?.max_context_length ?? status?.native_context_length ?? null;
  const ready = Boolean(status?.active_model) && status?.is_gguf !== false;
  const [setupOpen, setSetupOpen] = useState(true);
  const rowsOn = config.variants.filter(
    (v) => !disabled.includes(v.label),
  ).length;

  useEffect(() => {
    if (!runsLoaded) void refreshRuns();
  }, [runsLoaded, refreshRuns]);

  // The newest run stands in until one is picked; its measurements are fetched on demand.
  const shownId = selectedRunId ?? runs[0]?.id ?? null;
  useEffect(() => {
    if (shownId && !loaded[shownId]) void selectRun(shownId);
  }, [shownId, loaded, selectRun]);
  const shown = live ? live.run : shownId ? (loaded[shownId] ?? null) : null;

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <div className="mx-auto flex w-full max-w-[calc(1180px*var(--ui-space-scale,1))] flex-col gap-6 px-5 pb-20 pt-8 sm:px-9 sm:pt-10">
        <header className="font-heading flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
          <div className="flex flex-col gap-0.5">
            <h1 className="page-title-halo text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
              Benchmarks
            </h1>
            <p className="page-title-halo text-sm text-muted-foreground">
              Find the fastest settings for the model you have loaded, on this
              machine
            </p>
          </div>
          <div className="flex min-w-0 flex-wrap items-center justify-end gap-1.5 sm:flex-1">
            {machine.gpu && (
              <StatPill icon={DashboardSpeed01Icon} value={machine.gpu} />
            )}
            {machine.vramGb && (
              <StatPill
                icon={CpuIcon}
                value={`${Math.round(machine.vramGb)} GiB`}
                label="VRAM"
              />
            )}
            {machine.backend && (
              <StatPill
                value={machine.backend}
                label={machine.llamaTag ?? undefined}
              />
            )}
          </div>
        </header>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <ModelStrip status={status} run={live?.run ?? null} />
          <div className="flex items-center gap-2">
            <History shownId={shownId} />
            {!live && (
              <Button
                onClick={() => void start()}
                disabled={!ready || rowsOn === 0}
                className="h-9 gap-2 rounded-full px-4"
              >
                <HugeiconsIcon
                  icon={Rocket01Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
                {config.sweep === "tune" ? "Run auto-tune" : "Run benchmark"}
              </Button>
            )}
          </div>
        </div>

        {error && (
          <p
            role="alert"
            className="rounded-xl bg-destructive/10 px-4 py-2.5 text-ui-12p5 text-destructive"
          >
            {error}
          </p>
        )}

        <div
          className={cn(
            "grid grid-cols-1 items-start gap-6",
            setupOpen && "lg:grid-cols-[calc(300px*var(--ui-space-scale,1))_minmax(0,1fr)]",
          )}
        >
          {setupOpen && (
            <div className="lg:sticky lg:top-6">
              <SetupPanel
                status={status}
                maxContext={maxContext}
                locked={Boolean(live)}
                onCollapse={() => setSetupOpen(false)}
              />
            </div>
          )}
          <div className="flex min-w-0 flex-col gap-4">
            {!setupOpen && (
              <button
                type="button"
                onClick={() => setSetupOpen(true)}
                className="self-start rounded-full bg-muted/60 px-3.5 py-1.5 text-ui-12 text-muted-foreground transition-colors hover:text-foreground"
              >
                Show setup
              </button>
            )}
            {shown ? (
              <>
                {shown.config.sweep === "tune" && (
                  <TuneVerdictCard run={shown} />
                )}
                <RunResults
                  key={shown.id}
                  run={shown}
                  live={Boolean(live)}
                  progress={live?.progress}
                  onStop={cancel}
                />
              </>
            ) : shownId ? (
              <div className="corner-squircle flex items-center rounded-3xl ring-1 ring-border/60 justify-center bg-card px-6 py-16 text-ui-13 text-muted-foreground">
                Loading the run…
              </div>
            ) : (
              <div className="corner-squircle flex flex-col items-center rounded-3xl ring-1 ring-border/60 justify-center gap-1 bg-card px-6 py-16 text-center">
                <span className="text-ui-13p5 font-medium text-foreground">
                  No runs yet
                </span>
                <span className="text-ui-12 text-muted-foreground">
                  Pick a sweep and press Run. The chart fills in here as each
                  setting finishes.
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
