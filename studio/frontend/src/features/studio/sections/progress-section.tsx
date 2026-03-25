// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Progress } from "@/components/ui/progress";
import { OPTIMIZER_OPTIONS } from "@/config/training";
import { setTrainingCompareHandoff } from "@/features/chat";
import {
  useTrainingActions,
  useTrainingConfigStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import { useGpuUtilization } from "@/hooks";
import { cn } from "@/lib/utils";
import {
  ChartAverageIcon,
  DashboardSpeed01Icon,
  Notebook01Icon,
  RamMemoryIcon,
  StopIcon,
  TemperatureIcon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Link, useNavigate } from "@tanstack/react-router";
import { type ReactElement, type ReactNode, useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { ChartSettingsSheet } from "./charts/chart-settings-sheet";
import {
  formatDuration,
  formatNumber,
  phaseColors,
  phaseLabel,
} from "./progress-section-lib";

type ConfigGroup = {
  section: string;
  rows: [string, string | number | null | undefined][];
};

function configRow(
  label: string,
  value: string | number | null | undefined,
): [string, string | number | null | undefined] {
  return [label, value];
}

export function ProgressSection(): ReactElement {
  const navigate = useNavigate();
  const runtime = useTrainingRuntimeStore(
    useShallow((state) => ({
      phase: state.phase,
      message: state.message,
      error: state.error,
      currentStep: state.currentStep,
      totalSteps: state.totalSteps,
      currentEpoch: state.currentEpoch,
      currentLoss: state.currentLoss,
      currentLearningRate: state.currentLearningRate,
      currentGradNorm: state.currentGradNorm,
      progressPercent: state.progressPercent,
      elapsedSeconds: state.elapsedSeconds,
      etaSeconds: state.etaSeconds,
      currentNumTokens: state.currentNumTokens,
      isTrainingRunning: state.isTrainingRunning,
      lossHistory: state.lossHistory,
      lrHistory: state.lrHistory,
      gradNormHistory: state.gradNormHistory,
    })),
  );

  const config = useTrainingConfigStore(
    useShallow((state) => ({
      selectedModel: state.selectedModel,
      trainingMethod: state.trainingMethod,
      epochs: state.epochs,
      batchSize: state.batchSize,
      learningRate: state.learningRate,
      maxSteps: state.maxSteps,
      contextLength: state.contextLength,
      warmupSteps: state.warmupSteps,
      optimizerType: state.optimizerType,
      loraRank: state.loraRank,
      loraAlpha: state.loraAlpha,
      loraDropout: state.loraDropout,
      loraVariant: state.loraVariant,
    })),
  );

  const { stopTrainingRun } = useTrainingActions();
  const gpu = useGpuUtilization(runtime.isTrainingRunning);
  const [stopDialogOpen, setStopDialogOpen] = useState(false);
  const [stopRequested, setStopRequested] = useState(false);
  const gpuEntries =
    gpu.gpus.length > 0
      ? gpu.gpus
      : gpu.available
        ? [
            {
              index: 0,
              name: null,
              gpu_utilization_pct: gpu.gpu_utilization_pct,
              temperature_c: gpu.temperature_c,
              vram_used_gb: gpu.vram_used_gb,
              vram_total_gb: gpu.vram_total_gb,
              vram_utilization_pct: gpu.vram_utilization_pct,
              power_draw_w: gpu.power_draw_w,
              power_limit_w: gpu.power_limit_w,
              power_utilization_pct: gpu.power_utilization_pct,
            },
          ]
        : [];
  const gpuCountLabel =
    gpu.physical_gpu_count > 0
      ? `${gpu.physical_gpu_count} GPU${gpu.physical_gpu_count === 1 ? "" : "s"}`
      : gpuEntries.length > 0
        ? `${gpuEntries.length} GPU${gpuEntries.length === 1 ? "" : "s"}`
        : "No GPU";

  useEffect(() => {
    if (!runtime.isTrainingRunning) {
      setStopRequested(false);
    }
  }, [runtime.isTrainingRunning]);

  const pct =
    runtime.totalSteps > 0
      ? Math.min(
          100,
          Math.max(
            0,
            Math.round((runtime.currentStep / runtime.totalSteps) * 100),
          ),
        )
      : Math.round(runtime.progressPercent);

  const elapsed = runtime.elapsedSeconds;
  const derivedEta =
    elapsed != null && pct > 0
      ? Math.round((elapsed * (100 - pct)) / Math.max(pct, 1))
      : null;
  const eta = runtime.etaSeconds ?? derivedEta;

  const stepsPerSecond =
    elapsed != null && elapsed > 0 ? runtime.currentStep / elapsed : null;
  const showHalfwayHint =
    runtime.phase === "training" && pct >= 50 && pct < 100;
  const showCompletedHint = runtime.phase === "completed";
  const handleCompareInChat = async () => {
    setTrainingCompareHandoff(config.selectedModel);
    await navigate({ to: "/chat" });
  };
  const requestStop = async (saveCheckpoint: boolean) => {
    setStopRequested(true);
    setStopDialogOpen(false);
    useTrainingRuntimeStore.getState().setStopRequested(true);
    try {
      const ok = await stopTrainingRun(saveCheckpoint);
      if (!ok) {
        setStopRequested(false);
      }
    } catch {
      setStopRequested(false);
    }
  };

  const stoppedLoss = getDisplayMetric(
    runtime.isTrainingRunning,
    runtime.currentLoss,
    runtime.lossHistory,
  );
  const stoppedLr = getDisplayMetric(
    runtime.isTrainingRunning,
    runtime.currentLearningRate,
    runtime.lrHistory,
  );
  const stoppedGradNorm = runtime.isTrainingRunning
    ? runtime.currentGradNorm
    : (lastNonZeroValue(runtime.gradNormHistory) ?? runtime.currentGradNorm);

  const optimizerLabel =
    OPTIMIZER_OPTIONS.find((o) => o.value === config.optimizerType)?.label ??
    config.optimizerType;

  const configItems: ConfigGroup[] = [
    {
      section: "Hyperparams",
      rows: [
        configRow("Epochs", config.epochs),
        configRow("Batch size", config.batchSize),
        configRow("Learning rate", config.learningRate),
        configRow("Optimizer", optimizerLabel),
        configRow("Max steps", config.maxSteps),
        configRow("Context length", config.contextLength),
        configRow("Warmup steps", config.warmupSteps),
      ],
    },
    ...(config.trainingMethod !== "full"
      ? [
          {
            section: "LoRA",
            rows: [
              configRow("Rank", config.loraRank),
              configRow("Alpha", config.loraAlpha),
              configRow("Dropout", config.loraDropout),
              configRow("Variant", config.loraVariant),
            ],
          },
        ]
      : []),
  ];

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
      title="Training Progress"
      description={runtime.message || "Live training metrics"}
      accent="emerald"
      className="shadow-border border border-border/60 bg-card/90 ring-0 backdrop-blur-sm"
      headerAction={
        <TrainingHeaderActions
          configItems={configItems}
          isTrainingRunning={runtime.isTrainingRunning}
          onOpenStopDialog={setStopDialogOpen}
          onRequestStop={requestStop}
          stopDialogOpen={stopDialogOpen}
          stopRequested={stopRequested}
        />
      }
    >
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(18rem,0.8fr)]">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${phaseColors[runtime.phase]}`}
            >
              {phaseLabel[runtime.phase]}
            </span>
            <span className="text-[10px] tabular-nums text-muted-foreground">
              Epoch {runtime.currentEpoch.toFixed(2)}
            </span>
            <span className="rounded-full border border-border/60 px-2.5 py-1 text-[10px] font-medium tabular-nums text-muted-foreground">
              {pct}% complete
            </span>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                Step {runtime.currentStep} / {runtime.totalSteps || "--"}
              </span>
              <span>{pct}%</span>
            </div>
            <Progress value={pct} className="h-2 bg-foreground/[0.05]" />
          </div>

          <MilestoneCallout
            showCompletedHint={showCompletedHint}
            showHalfwayHint={showHalfwayHint}
            onCompareInChat={handleCompareInChat}
          />

          {runtime.error && (
            <p className="rounded-2xl border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-red-500 leading-relaxed">
              {runtime.error}
            </p>
          )}

          <div className="grid gap-x-4 gap-y-3 pt-1 sm:grid-cols-2 xl:grid-cols-5">
            <MetricStat
              label="Loss"
              valueClassName="text-2xl font-bold tracking-tight"
            >
              {stoppedLoss.toFixed(4)}
            </MetricStat>
            <MetricStat label="LR">{stoppedLr.toExponential(2)}</MetricStat>
            <MetricStat label="Grad Norm">
              {formatNumber(stoppedGradNorm, 3)}
            </MetricStat>
            <MetricStat label="Model" valueClassName="truncate">
              {config.selectedModel ?? "--"}
            </MetricStat>
            <MetricStat label="Method">
              {config.trainingMethod === "qlora" ? "QLoRA" : config.trainingMethod === "lora" ? "LoRA" : "Full"}
            </MetricStat>
          </div>

          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span>Elapsed: {formatDuration(elapsed)}</span>
            <span>ETA: {formatDuration(eta)}</span>
            <span>
              {stepsPerSecond == null
                ? "-- steps/s"
                : `${stepsPerSecond.toFixed(2)} steps/s`}
            </span>
            {runtime.currentNumTokens != null && (
              <span>Tokens: {runtime.currentNumTokens}</span>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs font-medium text-muted-foreground">
              GPU Monitor
            </p>
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
              <span>{gpuCountLabel}</span>
              <span>Live</span>
            </div>
          </div>
          <div
            className={cn(
              "grid gap-3",
              gpuEntries.length > 1
                ? "grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4"
                : "grid-cols-1",
            )}
          >
            {gpuEntries.map((entry) => (
              <GpuMonitorCard key={entry.index} gpu={entry} />
            ))}
          </div>
        </div>
      </div>
    </SectionCard>
  );
}

function TrainingHeaderActions({
  configItems,
  isTrainingRunning,
  onOpenStopDialog,
  onRequestStop,
  stopDialogOpen,
  stopRequested,
}: {
  configItems: ConfigGroup[];
  isTrainingRunning: boolean;
  onOpenStopDialog: (open: boolean) => void;
  onRequestStop: (saveCheckpoint: boolean) => Promise<void>;
  stopDialogOpen: boolean;
  stopRequested: boolean;
}): ReactElement {
  return (
    <div className="flex items-center gap-2">
      <Popover>
        <PopoverTrigger asChild={true}>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label="Open training config"
          >
            <HugeiconsIcon icon={Notebook01Icon} className="size-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-72" align="end">
          <div className="flex flex-col gap-3">
            <p className="text-xs font-semibold">Training Config</p>
            {configItems.map((group) => (
              <div key={group.section} className="flex flex-col gap-1">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  {group.section}
                </p>
                {group.rows.map(([label, value]) => (
                  <div key={label} className="flex justify-between text-xs">
                    <span className="text-muted-foreground">{label}</span>
                    <span className="font-medium tabular-nums">
                      {String(value)}
                    </span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </PopoverContent>
      </Popover>
      <ChartSettingsSheet />
      <AlertDialog open={stopDialogOpen} onOpenChange={onOpenStopDialog}>
        <Button
          data-tour="studio-training-stop"
          variant="destructive"
          size="sm"
          className={cn(
            "h-8 rounded-full px-3.5 text-xs shadow-sm",
            stopRequested ? "cursor-not-allowed opacity-60" : "cursor-pointer",
          )}
          onClick={() => onOpenStopDialog(true)}
          disabled={!isTrainingRunning || stopRequested}
        >
          <HugeiconsIcon icon={StopIcon} className="size-3" />
          {stopRequested ? "Stopping…" : "Stop"}
        </Button>
        <AlertDialogContent overlayClassName="bg-background/40 supports-backdrop-filter:backdrop-blur-[1px]">
          <AlertDialogHeader>
            <AlertDialogTitle>Stop Training</AlertDialogTitle>
            <AlertDialogDescription>
              Choose how you want to stop the current training run.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Continue Training</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => onRequestStop(false)}
            >
              Cancel Training
            </AlertDialogAction>
            <AlertDialogAction onClick={() => onRequestStop(true)}>
              Stop and Save
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function MilestoneCallout({
  showCompletedHint,
  showHalfwayHint,
  onCompareInChat,
}: {
  showCompletedHint: boolean;
  showHalfwayHint: boolean;
  onCompareInChat: () => Promise<void>;
}): ReactElement | null {
  if (!(showHalfwayHint || showCompletedHint)) {
    return null;
  }

  return (
    <div className="corner-squircle rounded-2xl border border-border/60 bg-muted/30 px-3 py-2.5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          {!showCompletedHint && (
            <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
              Milestone
            </p>
          )}
          <p
            className={cn(
              "text-xs text-foreground/85",
              !showCompletedHint && "mt-1",
            )}
          >
            {showCompletedHint
              ? "Training done. Next step: compare base vs fine-tuned outputs."
              : "Halfway done. Training is past 50%."}
          </p>
        </div>
        {!showCompletedHint && (
          <span className="rounded-full border border-border/60 bg-background/80 px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
            50%+
          </span>
        )}
      </div>
      {showCompletedHint && (
        <div className="mt-2 flex flex-wrap gap-2">
          <Button size="xs" onClick={onCompareInChat}>
            Compare in Chat
          </Button>
          <Button asChild={true} size="xs" variant="outline">
            <Link to="/export">Export Model</Link>
          </Button>
        </div>
      )}
    </div>
  );
}

function MetricStat({
  label,
  children,
  valueClassName,
}: {
  label: string;
  children: ReactNode;
  valueClassName?: string;
}): ReactElement {
  return (
    <div className="min-w-0">
      <p className="text-[11px] text-muted-foreground">{label}</p>
      <p
        className={`mt-1 text-base font-semibold tabular-nums ${valueClassName ?? ""}`}
      >
        {children}
      </p>
    </div>
  );
}

function lastNonZeroValue(points: { value: number }[]): number | null {
  for (let i = points.length - 1; i >= 0; i -= 1) {
    const value = points[i]?.value;
    if (Number.isFinite(value) && value !== 0) {
      return value;
    }
  }
  return null;
}

function getDisplayMetric(
  isTrainingRunning: boolean,
  currentValue: number,
  history: { value: number }[],
): number {
  if (isTrainingRunning) {
    return currentValue;
  }
  return lastNonZeroValue(history) ?? currentValue;
}

function GpuStat({
  label,
  icon,
  value,
  detail,
  pct,
  max,
}: {
  label: string;
  icon: ReactNode;
  value: string;
  detail?: string;
  pct: number;
  max?: number;
}): ReactElement {
  const clamped = Math.max(0, Math.min(pct, max ?? 100));
  let barColor = "bg-red-500";
  if (clamped < 60) {
    barColor = "bg-emerald-500";
  } else if (clamped < 95) {
    barColor = "bg-amber-500";
  }

  return (
    <div className="corner-squircle min-w-0 rounded-2xl border border-border/50 bg-background/60 p-3">
      <div className="flex min-w-0 items-center gap-1.5 text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
        <span className="shrink-0">
          {icon}
        </span>
        <span className="truncate">{label}</span>
      </div>
      <div className="mt-2 min-w-0">
        <p className="truncate text-sm font-semibold leading-none tabular-nums text-foreground">
          {value}
        </p>
        {detail ? (
          <p className="mt-1 truncate text-[11px] text-muted-foreground">
            {detail}
          </p>
        ) : null}
      </div>
      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-muted/80">
        <div
          className={`h-full rounded-full ${barColor} transition-all duration-300`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

function GpuMonitorCard({
  gpu,
}: {
  gpu: {
    index: number;
    name?: string | null;
    gpu_utilization_pct: number | null;
    temperature_c: number | null;
    vram_used_gb: number | null;
    vram_total_gb: number | null;
    vram_utilization_pct: number | null;
    power_draw_w: number | null;
    power_limit_w: number | null;
    power_utilization_pct: number | null;
  };
}): ReactElement {
  return (
    <div className="corner-squircle min-w-0 rounded-2xl border border-border/50 bg-background/40 p-3">
      <div className="mb-3 flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-medium">GPU {gpu.index}</p>
          <p className="mt-1 line-clamp-2 text-[11px] leading-4 text-muted-foreground">
            {gpu.name ?? "NVIDIA GPU"}
          </p>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        <GpuStat
          label="Utilization"
          icon={<HugeiconsIcon icon={DashboardSpeed01Icon} className="size-3.5" />}
          value={
            gpu.gpu_utilization_pct != null ? `${gpu.gpu_utilization_pct}%` : "--"
          }
          pct={gpu.gpu_utilization_pct ?? 0}
        />
        <GpuStat
          label="Temperature"
          icon={<HugeiconsIcon icon={TemperatureIcon} className="size-3.5" />}
          value={gpu.temperature_c != null ? `${gpu.temperature_c}°C` : "--"}
          detail={gpu.temperature_c != null ? "Sensor reading" : undefined}
          pct={gpu.temperature_c ?? 0}
          max={100}
        />
        <GpuStat
          label="VRAM"
          icon={<HugeiconsIcon icon={RamMemoryIcon} className="size-3.5" />}
          value={
            gpu.vram_used_gb != null ? `${gpu.vram_used_gb} GB used` : "--"
          }
          detail={
            gpu.vram_total_gb != null ? `${gpu.vram_total_gb} GB total` : undefined
          }
          pct={gpu.vram_utilization_pct ?? 0}
        />
        <GpuStat
          label="Power"
          icon={<HugeiconsIcon icon={ZapIcon} className="size-3.5" />}
          value={
            gpu.power_draw_w != null
              ? `${gpu.power_draw_w} W draw`
              : "--"
          }
          detail={
            gpu.power_limit_w != null ? `${gpu.power_limit_w} W limit` : undefined
          }
          pct={gpu.power_utilization_pct ?? 0}
        />
      </div>
    </div>
  );
}
