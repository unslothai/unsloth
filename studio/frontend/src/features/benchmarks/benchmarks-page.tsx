// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Benchmarks: sweep load settings on the loaded GGUF and chart what each one does to
// throughput, draft acceptance and load time on this machine. Setup, results and a
// Train-style run preview on one tab; saved runs on History.

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  type InferenceStatusResponse,
  fetchGgufStagedMetadata,
  getInferenceStatus,
} from "@/features/chat";
import { authFetch } from "@/features/auth";
import { gpuMemoryDisplay } from "@/hooks/gpu-memory-display";
import { gpuMemoryTotalsGb, resolveGpuVramUsedGb } from "@/hooks/gpu-vram";
import { useSystemInfo } from "@/hooks/use-system";
import { cn } from "@/lib/utils";
import {
  Chip02Icon,
  CpuIcon,
  GpuIcon,
  RamMemoryIcon,
} from "@hugeicons/core-free-icons";
import { type ReactElement, type ReactNode, useEffect, useState } from "react";
import { HistoryGrid } from "./components/history-grid";
import { RunResults } from "./components/results-panel";
import { RunPreviewCard } from "./components/run-preview";
import {
  BenchModelPicker,
  SetupPanel,
  StatPill,
} from "./components/setup-panel";
import { TuneVerdictCard } from "./components/tune-section";
import { useLocale } from "@/i18n";
import { ago } from "./lib/ago";
import {
  type BenchRun,
  type ModelShape,
  SWEEP_TITLE,
  modelShort,
} from "./lib/bench-math";
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

interface Backend {
  name: string | null;
  tag: string | null;
}

const BACKEND_NAME: Record<string, string> = {
  cuda: "CUDA",
  rocm: "ROCm",
  vulkan: "Vulkan",
  metal: "Metal",
  cpu: "CPU",
  sycl: "SYCL",
};

function useLlamaBackend(): Backend {
  const [b, setB] = useState<Backend>({ name: null, tag: null });
  useEffect(() => {
    void authFetch("/api/llama/backend")
      .then((res) => (res.ok ? res.json() : null))
      .then((llama: Record<string, unknown> | null) => {
        const name = typeof llama?.backend === "string" ? llama.backend : null;
        setB({
          name: name ? (BACKEND_NAME[name.toLowerCase()] ?? name) : null,
          tag:
            typeof llama?.installed_tag === "string"
              ? llama.installed_tag
              : null,
        });
      })
      .catch(() => undefined);
  }, []);
  return b;
}

const gb = (n: number) => (n >= 10 ? Math.round(n) : Math.round(n * 10) / 10);

/** GPUs, VRAM, RAM, CPU and llama.cpp backend. Polls while idle; a run holds the last
 * reading so the probe doesn't compete with what's being measured. */
function MachinePills({ polling }: { polling: boolean }): ReactElement {
  const sys = useSystemInfo({ pollMs: polling ? 5000 : 0 });
  const backend = useLlamaBackend();
  // llama.cpp's device list, which differs from torch's when it runs on Vulkan.
  const gpu = sys.inference_gpu?.available ? sys.inference_gpu : sys.gpu;
  const display = gpuMemoryDisplay(gpu);
  const devices = display.sharedOnly
    ? display.sharedDevices
    : display.usageDevices;
  const names = devices.map((d) => d.name ?? "GPU");
  const vramTotal = gpuMemoryTotalsGb(devices).total;
  const vramUsed = display.sharedOnly
    ? null
    : resolveGpuVramUsedGb(display.usageGpu);
  const ramTotal = sys.memory.total_gb;
  const ramUsed = Math.max(0, ramTotal - sys.memory.available_gb);
  const threads = sys.cpu.logical_count;

  const gpuLabel =
    names.length === 0
      ? null
      : new Set(names).size === 1
        ? names.length > 1
          ? `${names.length}× ${names[0]}`
          : names[0]
        : `${names[0]} +${names.length - 1}`;
  const perDevice = devices
    .map(
      (d, i) =>
        `GPU ${d.index ?? i}: ${d.name ?? "GPU"}${
          d.memory_total_gb
            ? ` · ${d.vram_used_gb != null ? `${gb(d.vram_used_gb)} / ` : ""}${gb(d.memory_total_gb)} GB`
            : ""
        }`,
    )
    .join("\n");

  return (
    <div className="flex min-w-0 flex-wrap items-center justify-end gap-1.5 sm:flex-1">
      {gpuLabel && (
        <StatPill icon={GpuIcon} value={gpuLabel} title={perDevice} />
      )}
      {vramTotal > 0 && (
        <StatPill
          icon={Chip02Icon}
          value={
            vramUsed != null
              ? `${gb(vramUsed)} / ${gb(vramTotal)} GB`
              : `${gb(vramTotal)} GB`
          }
          label={display.sharedOnly ? "shared" : "VRAM"}
          title={perDevice}
        />
      )}
      {ramTotal > 0 && (
        <StatPill
          icon={RamMemoryIcon}
          value={`${gb(ramUsed)} / ${gb(ramTotal)} GB`}
          label="RAM"
        />
      )}
      {threads > 0 && (
        <StatPill icon={CpuIcon} value={threads} label="threads" />
      )}
      {backend.name && (
        <StatPill
          value={backend.name}
          label={backend.tag ? `llama.cpp ${backend.tag}` : "llama.cpp"}
        />
      )}
    </div>
  );
}

/** One quiet line over a finished run: which run this is, and whether the setup has moved on. */
function ShownRunNote({
  run,
  latest,
  changed,
  onLatest,
}: {
  run: BenchRun;
  latest: boolean;
  changed: string | null;
  onLatest: (() => void) | null;
}): ReactElement {
  const locale = useLocale();
  return (
    <div className="flex min-h-7 flex-wrap items-center gap-x-2 gap-y-1 px-1 text-ui-12 text-muted-foreground duration-300 animate-in fade-in-0">
      <span
        className="size-1.5 shrink-0 rounded-full bg-muted-foreground/50"
        aria-hidden={true}
      />
      <span>
        {latest ? "Your last run" : "A saved run"},{" "}
        <span title={new Date(run.createdAt).toLocaleString()}>
          {ago(run.createdAt, locale)}
        </span>
      </span>
      {changed && (
        <span className="min-w-0 truncate text-muted-foreground/80">
          · Run to measure {changed}
        </span>
      )}
      {onLatest && (
        <button
          type="button"
          onClick={onLatest}
          className="ml-auto rounded-full px-2.5 py-0.5 transition-colors hover:bg-muted/60 hover:text-foreground"
        >
          Back to the latest
        </button>
      )}
    </div>
  );
}

type BenchTab = "benchmark" | "history";

/** Train's sub-nav: underlined text tabs on the header rule. */
function BenchSubNav({
  value,
  runCount,
}: {
  value: BenchTab;
  runCount: number;
}): ReactElement {
  const items: ReadonlyArray<{
    value: BenchTab;
    label: ReactNode;
    disabled: boolean;
  }> = [
    { value: "benchmark", label: "Benchmark", disabled: false },
    {
      value: "history",
      label: (
        <>
          History
          {runCount > 0 && (
            <span className="ml-1.5 font-normal tabular-nums text-muted-foreground">
              {runCount}
            </span>
          )}
        </>
      ),
      disabled: false,
    },
  ];
  return (
    <TabsList
      unstyled={true}
      className="flex min-w-0 flex-wrap items-center justify-start gap-3 pb-px text-ui-13 tracking-nav sm:gap-6"
    >
      {items.map((item) => {
        const active = value === item.value;
        return (
          <TabsTrigger
            key={item.value}
            value={item.value}
            disabled={item.disabled}
            indicatorClassName="hidden"
            className={cn(
              "relative h-9 flex-none select-none rounded-none border-0 px-0 py-0 text-ui-13 transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              "after:pointer-events-none after:absolute after:inset-x-0 after:bottom-0 after:h-[2px] after:rounded-full after:bg-foreground after:transition-opacity",
              active
                ? "font-semibold text-foreground after:opacity-100"
                : "text-muted-foreground hover:text-foreground after:opacity-0",
            )}
          >
            {item.label}
          </TabsTrigger>
        );
      })}
    </TabsList>
  );
}

/** Layer counts (offload sweep) and context window (context sweep): the loaded model's from
 * status, a picked one's from its GGUF header. */
function useModelShape(
  status: InferenceStatusResponse | null,
  model: string | null,
  variant: string | null,
): { shape: ModelShape | null; contextLength: number | null } {
  const [picked, setPicked] = useState<{
    key: string;
    shape: ModelShape;
    contextLength: number | null;
  } | null>(null);
  const key = model ? `${model}\u0000${variant ?? ""}` : null;
  useEffect(() => {
    if (!model || !key) return;
    let cancelled = false;
    void fetchGgufStagedMetadata({ model_path: model, gguf_variant: variant })
      .then((m) => {
        if (!cancelled)
          setPicked({
            key,
            shape: { layers: m.layerCount, moeLayers: m.moeLayerCount },
            contextLength: m.contextLength,
          });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [model, variant, key]);
  if (key) {
    const p = picked?.key === key ? picked : null;
    return { shape: p?.shape ?? null, contextLength: p?.contextLength ?? null };
  }
  if (!status?.active_model) return { shape: null, contextLength: null };
  return {
    shape: {
      layers: status.n_layers ?? null,
      moeLayers: status.n_moe_layers ?? null,
    },
    contextLength: null,
  };
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
  const status = useLoadedModel(Boolean(live));
  const config = useBenchmarksStore((s) => s.config);
  const choosePreset = useBenchmarksStore((s) => s.choosePreset);
  const { shape, contextLength: pickedContext } = useModelShape(
    status,
    config.tuneModel ?? null,
    config.tuneVariant ?? null,
  );
  // A picked model sweeps its own context window; while its header loads, fall back to chat's.
  const residentContext =
    status?.max_context_length ?? status?.native_context_length ?? null;
  const maxContext = config.tuneModel
    ? (pickedContext ?? residentContext)
    : residentContext;
  // The offload rows are scaled to the model, so a new model or a late shape rebuilds them.
  const shapeKey = shape ? `${shape.layers}/${shape.moeLayers}` : "";
  const offloadSweep = config.sweep === "offload";
  useEffect(() => {
    if (offloadSweep && !live && shapeKey)
      choosePreset("offload", maxContext, shape);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the shape's value
  }, [offloadSweep, shapeKey]);
  const [tab, setTab] = useState<BenchTab>("benchmark");

  useEffect(() => {
    if (!runsLoaded) void refreshRuns();
  }, [runsLoaded, refreshRuns]);

  // The newest run stands in until one is picked; its measurements are fetched on demand.
  const shownId = selectedRunId ?? runs[0]?.id ?? null;
  useEffect(() => {
    if (shownId && !loaded[shownId]) void selectRun(shownId);
  }, [shownId, loaded, selectRun]);
  const shown = live ? live.run : shownId ? (loaded[shownId] ?? null) : null;
  const latestId = runs[0]?.id ?? null;
  // What Run would measure now, named when it isn't what the chart shows.
  const nextModel = modelShort(config.tuneModel ?? status?.active_model ?? "");
  const changed =
    !shown || live
      ? null
      : config.sweep !== shown.config.sweep
        ? `${SWEEP_TITLE[config.sweep]}${nextModel ? ` on ${nextModel}` : ""}`
        : nextModel && nextModel !== modelShort(shown.model)
          ? nextModel
          : null;

  const preview = (
    <RunPreviewCard
      status={status}
      onRun={() => void start()}
      onViewRun={() => setTab("benchmark")}
    />
  );

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <Tabs
        value={tab}
        onValueChange={(v) => setTab(v as BenchTab)}
        className="contents"
      >
        <div className="mx-auto flex w-full max-w-[calc(1180px*var(--ui-space-scale,1))] 3xl:max-w-[calc(1440px*var(--ui-space-scale,1))] 4xl:max-w-[calc(1760px*var(--ui-space-scale,1))] flex-col gap-7 px-5 pb-20 pt-8 max-sm:px-4 sm:px-9 sm:pt-10">
          <header className="font-heading flex flex-col gap-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
              <div className="flex flex-col gap-0.5">
                <h1 className="page-title-halo text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
                  Benchmarks
                </h1>
                <p className="page-title-halo text-sm text-muted-foreground">
                  Find the fastest settings for the model you have loaded, on
                  this machine
                </p>
              </div>
              <MachinePills polling={!live} />
            </div>
            <div className="flex min-w-0 flex-wrap items-center gap-3 border-b border-border/60">
              <BenchSubNav value={tab} runCount={runs.length} />
              <div className="ml-auto min-w-0 max-w-full pb-1.5 sm:max-w-[60%]">
                <BenchModelPicker status={status} locked={Boolean(live)} />
              </div>
            </div>
          </header>

          {error && (
            <p
              role="alert"
              className="rounded-xl bg-destructive/10 px-4 py-2.5 text-ui-12p5 text-destructive"
            >
              {error}
            </p>
          )}

          <TabsContent value="benchmark" className="mt-0">
            {/* Setup, the chart, and the run preview side by side. Below 72rem the preview
                moves under the setup, in the same column, so nothing jumps between rows. */}
            <div className="@container/bench">
              <div className="grid grid-cols-1 items-start gap-6 @3xl/bench:grid-cols-[calc(264px*var(--ui-space-scale,1))_minmax(0,1fr)] @6xl/bench:grid-cols-[calc(264px*var(--ui-space-scale,1))_minmax(0,1fr)_calc(264px*var(--ui-space-scale,1))]">
                <div className="flex min-w-0 flex-col gap-6 @3xl/bench:col-start-1 @3xl/bench:row-start-1 @6xl/bench:sticky @6xl/bench:top-6">
                  <SetupPanel
                    maxContext={maxContext}
                    shape={shape}
                    locked={Boolean(live)}
                  />
                  <div className="@6xl/bench:hidden">{preview}</div>
                </div>
                <div className="hidden @6xl/bench:sticky @6xl/bench:top-6 @6xl/bench:col-start-3 @6xl/bench:row-start-1 @6xl/bench:block">
                  {preview}
                </div>
                <div className="flex min-w-0 flex-col gap-4 @3xl/bench:col-start-2 @3xl/bench:row-start-1">
                  {shown ? (
                    <div
                      key={shown.id}
                      className="flex flex-col gap-4 duration-300 animate-in fade-in-0"
                    >
                      {!live && (
                        <ShownRunNote
                          run={shown}
                          latest={shown.id === latestId}
                          changed={changed}
                          onLatest={
                            latestId && shown.id !== latestId
                              ? () => void selectRun(latestId)
                              : null
                          }
                        />
                      )}
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
                    </div>
                  ) : shownId ? (
                    <div className="corner-squircle flex items-center justify-center rounded-3xl bg-card px-6 py-16 text-ui-13 text-muted-foreground ring-1 ring-border/60">
                      Loading the run…
                    </div>
                  ) : (
                    <div className="corner-squircle flex flex-col items-center justify-center gap-1 rounded-3xl bg-card px-6 py-16 text-center ring-1 ring-border/60">
                      <span className="text-ui-13p5 font-medium text-foreground">
                        No runs yet
                      </span>
                      <span className="text-ui-12 text-muted-foreground">
                        Pick a sweep and press Run. The chart fills in here as
                        each setting finishes.
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="history" className="mt-0">
            <HistoryGrid
              onOpen={(id) => {
                void selectRun(id);
                setTab("benchmark");
              }}
            />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
