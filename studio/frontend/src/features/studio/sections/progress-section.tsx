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
import type { TrainingViewData } from "@/features/training";
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
import { type ReactElement, type ReactNode, useState } from "react";
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

interface ProgressSectionProps {
  data: TrainingViewData;
  isHistorical?: boolean;
  configOverride?: {
    epochs?: number;
    batchSize?: number;
    learningRate?: string;
    maxSteps?: number;
    contextLength?: number;
    warmupSteps?: number;
    optimizerType?: string;
    loraRank?: number;
    loraAlpha?: number;
    loraDropout?: number;
    loraVariant?: string;
  };
}

export function ProgressSection({
  data,
  isHistorical = false,
  configOverride,
}: ProgressSectionProps): ReactElement {
  const navigate = useNavigate();

  const config = useTrainingConfigStore(
    useShallow((state) => ({
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

  const [stopDialogOpen, setStopDialogOpen] = useState(false);
  const [stopRequestedLocal, setStopRequestedLocal] = useState(false);

  // Auto-reset when training stops -- no useEffect needed
  const stopRequested = data.isTrainingRunning && stopRequestedLocal;

  const pct =
    data.totalSteps > 0
      ? Math.min(
          100,
          Math.max(
            0,
            Math.round((data.currentStep / data.totalSteps) * 100),
          ),
        )
      : Math.round(data.progressPercent);

  const elapsed = data.elapsedSeconds;
  const derivedEta =
    elapsed != null && pct > 0
      ? Math.round((elapsed * (100 - pct)) / Math.max(pct, 1))
      : null;
  const eta = data.etaSeconds ?? derivedEta;

  const stepsPerSecond =
    elapsed != null && elapsed > 0 ? data.currentStep / elapsed : null;
  const showHalfwayHint =
    data.phase === "training" && pct >= 50 && pct < 100;
  const showCompletedHint = data.phase === "completed";
  const handleCompareInChat = async () => {
    setTrainingCompareHandoff(data.modelName);
    await navigate({ to: "/chat" });
  };

  const stoppedLoss = getDisplayMetric(
    data.isTrainingRunning,
    data.currentLoss,
    data.lossHistory,
  );
  const stoppedLr = getDisplayMetric(
    data.isTrainingRunning,
    data.currentLearningRate,
    data.lrHistory,
  );
  const stoppedGradNorm = data.isTrainingRunning
    ? data.currentGradNorm
    : (lastValue(data.gradNormHistory) ?? data.currentGradNorm);

  const cfgEpochs = isHistorical ? configOverride?.epochs : config.epochs;
  const cfgBatchSize = isHistorical ? configOverride?.batchSize : config.batchSize;
  const cfgLearningRate = isHistorical ? configOverride?.learningRate : config.learningRate;
  const cfgMaxSteps = isHistorical ? configOverride?.maxSteps : config.maxSteps;
  const cfgContextLength = isHistorical ? configOverride?.contextLength : config.contextLength;
  const cfgWarmupSteps = isHistorical ? configOverride?.warmupSteps : config.warmupSteps;
  const cfgOptimizerType = isHistorical ? configOverride?.optimizerType : config.optimizerType;
  const cfgLoraRank = isHistorical ? configOverride?.loraRank : config.loraRank;
  const cfgLoraAlpha = isHistorical ? configOverride?.loraAlpha : config.loraAlpha;
  const cfgLoraDropout = isHistorical ? configOverride?.loraDropout : config.loraDropout;
  const cfgLoraVariant = isHistorical ? configOverride?.loraVariant : config.loraVariant;

  const optimizerLabel =
    OPTIMIZER_OPTIONS.find((o) => o.value === cfgOptimizerType)?.label ??
    cfgOptimizerType;

  const configItems: ConfigGroup[] = [
    {
      section: "Hyperparams",
      rows: [
        configRow("Epochs", cfgEpochs),
        configRow("Batch size", cfgBatchSize),
        configRow("Learning rate", cfgLearningRate),
        configRow("Optimizer", optimizerLabel),
        configRow("Max steps", cfgMaxSteps),
        configRow("Context length", cfgContextLength),
        configRow("Warmup steps", cfgWarmupSteps),
      ],
    },
    ...(data.trainingMethod !== "full"
      ? [
          {
            section: "LoRA",
            rows: [
              configRow("Rank", cfgLoraRank),
              configRow("Alpha", cfgLoraAlpha),
              configRow("Dropout", cfgLoraDropout),
              configRow("Variant", cfgLoraVariant),
            ],
          },
        ]
      : []),
  ];

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
      title="Training Progress"
      description={data.message || "Live training metrics"}
      accent="emerald"
      className="shadow-border border border-border/60 bg-card/90 ring-0 backdrop-blur-sm"
      headerAction={
        isHistorical ? (
          <ConfigPopoverButton configItems={configItems} />
        ) : (
          <LiveTrainingHeaderActions
            configItems={configItems}
            isTrainingRunning={data.isTrainingRunning}
            onOpenStopDialog={setStopDialogOpen}
            stopDialogOpen={stopDialogOpen}
            stopRequested={stopRequested}
            onSetStopRequested={setStopRequestedLocal}
          />
        )
      }
    >
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(18rem,0.8fr)]">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${phaseColors[data.phase]}`}
            >
              {phaseLabel[data.phase]}
            </span>
            <span className="text-[10px] tabular-nums text-muted-foreground">
              Epoch {formatNumber(data.currentEpoch, 2)}
            </span>
            <span className="rounded-full border border-border/60 px-2.5 py-1 text-[10px] font-medium tabular-nums text-muted-foreground">
              {pct}% complete
            </span>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                Step {data.currentStep} / {data.totalSteps || "--"}
              </span>
              <span>{pct}%</span>
            </div>
            <Progress value={pct} className="h-2 bg-foreground/[0.05]" />
          </div>

          {!isHistorical && (
            <MilestoneCallout
              showCompletedHint={showCompletedHint}
              showHalfwayHint={showHalfwayHint}
              onCompareInChat={handleCompareInChat}
            />
          )}

          {data.error && (
            <p className="rounded-2xl border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-red-500 leading-relaxed">
              {data.error}
            </p>
          )}

          <div className="grid gap-x-4 gap-y-3 pt-1 sm:grid-cols-2 xl:grid-cols-5">
            <MetricStat
              label="Loss"
              valueClassName="text-2xl font-bold tracking-tight"
            >
              {stoppedLoss != null ? stoppedLoss.toFixed(4) : "--"}
            </MetricStat>
            <MetricStat label="LR">{stoppedLr != null ? stoppedLr.toExponential(2) : "--"}</MetricStat>
            <MetricStat label="Grad Norm">
              {formatNumber(stoppedGradNorm, 3)}
            </MetricStat>
            <MetricStat label="Model" valueClassName="truncate">
              {data.modelName || "--"}
            </MetricStat>
            <MetricStat label="Method">
              {data.trainingMethod === "qlora" ? "QLoRA" : data.trainingMethod === "lora" ? "LoRA" : "Full"}
            </MetricStat>
          </div>

          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span>Elapsed: {formatDuration(elapsed)}</span>
            {!isHistorical && <span>ETA: {formatDuration(eta)}</span>}
            <span>
              {stepsPerSecond == null
                ? "-- steps/s"
                : `${stepsPerSecond.toFixed(2)} steps/s`}
            </span>
            {data.currentNumTokens != null && (
              <span>Tokens: {data.currentNumTokens}</span>
            )}
          </div>
        </div>

        {!isHistorical && (
          <LiveGpuPanel isTrainingRunning={data.isTrainingRunning} />
        )}
      </div>
    </SectionCard>
  );
}

function LiveGpuPanel({
  isTrainingRunning,
}: {
  isTrainingRunning: boolean;
}): ReactElement {
  const gpu = useGpuUtilization(isTrainingRunning);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-muted-foreground">
          GPU Monitor
        </p>
        <span className="text-[11px] text-muted-foreground">Live</span>
      </div>
      <div className="grid grid-cols-2 gap-2.5">
        <GpuStat
          label="Utilization"
          icon={
            <HugeiconsIcon
              icon={DashboardSpeed01Icon}
              className="size-3.5"
            />
          }
          value={
            gpu.gpu_utilization_pct != null
              ? `${gpu.gpu_utilization_pct}%`
              : "--"
          }
          pct={gpu.gpu_utilization_pct ?? 0}
        />
        <GpuStat
          label="Temperature"
          icon={
            <HugeiconsIcon icon={TemperatureIcon} className="size-3.5" />
          }
          value={
            gpu.temperature_c != null ? `${gpu.temperature_c}°C` : "--"
          }
          pct={gpu.temperature_c ?? 0}
          max={100}
        />
        <GpuStat
          label="VRAM"
          icon={<HugeiconsIcon icon={RamMemoryIcon} className="size-3.5" />}
          value={
            gpu.vram_used_gb != null && gpu.vram_total_gb != null
              ? `${gpu.vram_used_gb} / ${gpu.vram_total_gb} GB`
              : "--"
          }
          pct={gpu.vram_utilization_pct ?? 0}
        />
        <GpuStat
          label="Power"
          icon={<HugeiconsIcon icon={ZapIcon} className="size-3.5" />}
          value={
            gpu.power_draw_w != null
              ? gpu.power_limit_w != null
                ? `${gpu.power_draw_w} / ${gpu.power_limit_w} W`
                : `${gpu.power_draw_w} W`
              : "--"
          }
          pct={gpu.power_utilization_pct ?? 0}
        />
      </div>
    </div>
  );
}

function LiveTrainingHeaderActions({
  configItems,
  isTrainingRunning,
  onOpenStopDialog,
  stopDialogOpen,
  stopRequested,
  onSetStopRequested,
}: {
  configItems: ConfigGroup[];
  isTrainingRunning: boolean;
  onOpenStopDialog: (open: boolean) => void;
  stopDialogOpen: boolean;
  stopRequested: boolean;
  onSetStopRequested: (v: boolean) => void;
}): ReactElement {
  const { stopTrainingRun } = useTrainingActions();

  const requestStop = async (saveCheckpoint: boolean) => {
    onSetStopRequested(true);
    onOpenStopDialog(false);
    useTrainingRuntimeStore.getState().setStopRequested(true);
    try {
      const ok = await stopTrainingRun(saveCheckpoint);
      if (!ok) {
        onSetStopRequested(false);
      }
    } catch {
      onSetStopRequested(false);
    }
  };

  return (
    <TrainingHeaderActions
      configItems={configItems}
      isTrainingRunning={isTrainingRunning}
      onOpenStopDialog={onOpenStopDialog}
      onRequestStop={requestStop}
      stopDialogOpen={stopDialogOpen}
      stopRequested={stopRequested}
    />
  );
}

function ConfigPopoverButton({
  configItems,
}: {
  configItems: ConfigGroup[];
}): ReactElement {
  return (
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
                    {value == null || value === "" ? "--" : String(value)}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
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
      <ConfigPopoverButton configItems={configItems} />
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

function lastValue(points: { value: number }[]): number | null {
  if (points.length === 0) return null;
  const v = points[points.length - 1]?.value;
  return v != null && Number.isFinite(v) ? v : null;
}

function getDisplayMetric(
  isTrainingRunning: boolean,
  currentValue: number | null,
  history: { value: number }[],
): number | null {
  if (isTrainingRunning) {
    return currentValue != null ? currentValue : null;
  }
  return lastValue(history) ?? (currentValue != null ? currentValue : null);
}

function GpuStat({
  label,
  icon,
  value,
  pct,
  max,
}: {
  label: string;
  icon: ReactNode;
  value: string;
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
    <div className="corner-squircle flex flex-col gap-2 rounded-2xl border border-border/50 bg-background/60 p-3">
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          {icon}
          {label}
        </span>
        <span className="font-medium tabular-nums">{value}</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted/80">
        <div
          className={`h-full rounded-full ${barColor} transition-all duration-300`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}
