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
import { useTranslation } from "react-i18next";
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
import { useCallback, useMemo } from "react";
import type { ReactElement } from "react";

function formatDuration(durationSeconds: number) {
  const seconds = Math.floor(durationSeconds);
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hrs > 0) {
    return `${hrs}h ${mins}m ${secs}s`;
  }
  if (mins > 0) {
    return `${mins}m ${secs}s`;
  }
  return `${secs}s`;
}

function formatNumber(n: number, d = 0): string {
  return n.toLocaleString(undefined, {
    maximumFractionDigits: d,
  });
}

type Phase = "loading" | "training" | "saving" | "completed" | "error" | "canceled";

const phaseLabel: Record<Phase, string> = {
  loading: "Loading",
  training: "Training",
  saving: "Saving checkpoint",
  completed: "Done",
  error: "Error",
  canceled: "Canceled",
};

const phaseColors: Record<Phase, string> = {
  loading: "bg-blue-500/15 text-blue-700 dark:text-blue-400",
  training: "bg-orange-500/15 text-orange-700 dark:text-orange-400",
  saving: "bg-amber-500/15 text-amber-700 dark:text-amber-400",
  completed: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  error: "bg-red-500/15 text-red-700 dark:text-red-400",
  canceled: "bg-muted",
};

function configRow<T>(label: string, value: T): [string, T] {
  return [label, value];
}

type ConfigGroup = { section: string; rows: Array<[string, unknown]> };

interface ProgressSectionProps {
  data: TrainingViewData;
  isHistorical?: boolean;
  configOverride?: {
    epochs?: number;
    batchSize?: number;
    learningRate?: number;
    optimizer?: string;
    maxSteps?: number;
    contextLength?: number;
    warmupSteps?: number;
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
  const { t } = useTranslation();
  const navigate = useNavigate();

  const config = useTrainingConfigStore(
    useShallow((state) => ({
      epochs: state.epochs,
      batchSize: state.batchSize,
      learningRate: state.learningRate,
      maxSteps: state.maxSteps,
      contextLength: state.contextLength,
      warmupSteps: state.warmupSteps,
      optimizer: state.optimizer,
    })),
  );

  const pct =
    data.totalSteps > 0
      ? Math.min(
          Math.round((data.currentStep / data.totalSteps) * 100),
          100,
        )
      : 0;

  const stoppedLoss = getDisplayMetric(
    data.isTrainingRunning,
    data.currentLoss,
    data.lossHistory,
  );

  const stoppedLr = data.isTrainingRunning
    ? data.currentLearningRate
    : lastValue(data.learningRateHistory) ?? data.currentLearningRate;

  const stoppedGradNorm = data.isTrainingRunning
    ? data.currentGradNorm
    : (lastValue(data.gradNormHistory) ?? data.currentGradNorm);

  const eta = data.etaSeconds;

  const elapsed = data.elapsedSeconds;

  const cfgEpochs = isHistorical ? configOverride?.epochs : config.epochs;
  const cfgBatchSize = isHistorical ? configOverride?.batchSize : config.batchSize;
  const cfgLearningRate = isHistorical ? configOverride?.learningRate : config.learningRate;
  const cfgMaxSteps = isHistorical ? configOverride?.maxSteps : config.maxSteps;
  const cfgContextLength = isHistorical ? configOverride?.contextLength : config.contextLength;
  const cfgWarmupSteps = isHistorical ? configOverride?.warmupSteps : config.warmupSteps;
  const cfgLoraRank = isHistorical ? configOverride?.loraRank : undefined;
  const cfgLoraAlpha = isHistorical ? configOverride?.loraAlpha : undefined;
  const cfgLoraDropout = isHistorical ? configOverride?.loraDropout : undefined;
  const cfgLoraVariant = isHistorical ? configOverride?.loraVariant : undefined;

  const optimizerMap: Record<string, string> = Object.fromEntries(OPTIMIZER_OPTIONS.map(o => o.value));
  const optimizerLabel =
    cfgLearningRate != null
      ? (optimizerMap[config.optimizer] ?? config.optimizer)
      : undefined;

  const [stopDialogOpen, setStopDialogOpen] = useState(false);

  const stopRequested = useTrainingRuntimeStore((s) => s.stopRequested);

  const [showCompletedHint, setShowCompletedHint] = useState(false);
  const [showHalfwayHint, setShowHalfwayHint] = useState(true);

  const [setStopRequestedLocal] = useState(false);

  const handleCompareInChat = useCallback(async () => {
    await setTrainingCompareHandoff(data.runId, data.modelName);
    navigate({ to: "/chat" });
  }, [data.runId, data.modelName, navigate]);

  useEffect(() => {
    if (data.phase !== "training") setShowHalfwayHint(false);
    else if (!showHalfwayHint && pct >= 50) setShowHalfwayHint(true);
  }, [data.phase, pct, showHalfwayHint]);

  useEffect(() => {
    if (data.phase === "completed" && !showCompletedHint) {
      setShowCompletedHint(true);
    }
  }, [data.phase, showCompletedHint]);

  const configItems: ConfigGroup[] = [
    {
      section: t("training.training"),
      rows: [
        configRow(t("training.epochs"), cfgEpochs),
        configRow(t("training.batchSize"), cfgBatchSize),
        configRow(t("training.learningRate"), cfgLearningRate),
        configRow(t("training.optimizer"), optimizerLabel),
        configRow(t("training.maxSteps"), cfgMaxSteps),
        configRow(t("training.contextLength"), cfgContextLength),
        configRow(t("training.warmupSteps"), cfgWarmupSteps),
      ],
    },
    ...(data.trainingMethod !== "full"
      ? [
          {
            section: "LoRA",
            rows: [
              configRow(t("training.rank"), cfgLoraRank),
              configRow(t("training.alpha"), cfgLoraAlpha),
              configRow(t("training.dropout"), cfgLoraDropout),
              configRow(t("training.variant"), cfgLoraVariant),
            ],
          },
        ]
      : []),
  ];

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
      title={t("training.trainingProgress")}
      description={data.message || t("training.liveMetrics")}
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
              {t("training.epoch")} {formatNumber(data.currentEpoch, 2)}
            </span>
            <span className="rounded-full border border-border/60 px-2.5 py-1 text-[10px] font-medium tabular-nums text-muted-foreground">
              {pct}% {t("training.complete")}
            </span>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                {t("training.step")} {data.currentStep} / {data.totalSteps || "--"}
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

          <div className="grid gap-x-4 gap-y-3 pt-1 sm:grid-cols-2 xl:grid-cols-5">
            <MetricStat label={t("training.loss")} valueClassName="text-2xl font-bold tracking-tight">
              {stoppedLoss != null ? stoppedLoss.toFixed(4) : "--"}
            </MetricStat>
            <MetricStat label={t("training.lr")}>{stoppedLr != null ? stoppedLr.toExponential(2) : "--"}</MetricStat>
            <MetricStat label={t("training.gradNorm")}>
              {formatNumber(stoppedGradNorm, 3)}
            </MetricStat>
            <MetricStat label={t("studio.model")} className="truncate">
              {data.modelName || "--"}
            </MetricStat>
            <MetricStat label={t("training.method")}>
              {data.trainingMethod === "qlora" ? t("training.qlora") : data.trainingMethod === "lora" ? t("training.lora") : t("training.full")}
            </MetricStat>
          </div>

          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span>{t("training.elapsed")}: {formatDuration(elapsed)}</span>
            {!isHistorical && <span>{t("training.eta")}: {formatDuration(eta)}</span>}
            <span
              className="rounded-full border border-border/60 bg-background/60 px-2.5 py-1 font-medium text-foreground"
            >
              {data.gpuInfo ?? "NVIDIA CUDA"}
            </span>
          </div>
        </div>
        <GpuMonitor isTrainingRunning={data.isTrainingRunning} />
      </div>
    </SectionCard>
  );
}

const GpuMonitor: FC<{ isTrainingRunning: boolean }> = ({ isTrainingRunning }) => {
  const gpu = useGpuUtilization(isTrainingRunning);
  const { t } = useTranslation();

  return (
    <div>
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-muted-foreground">
          {t("studio.gpuMonitor")}
        </p>
        <span className="text-[11px] text-muted-foreground">{t("studio.live")}</span>
      </div>
      <div className="grid grid-cols-2 gap-2.5">
        <GpuStat
          label={t("studio.utilization")}
          icon={<HugeiconsIcon icon={DashboardSpeed01Icon} className="size-3.5" />}
          value={gpu.gpu_utilization_pct != null ? `${gpu.gpu_utilization_pct}%` : "--"}
          pct={gpu.gpu_utilization_pct ?? 0}
        />
        <GpuStat
          label={t("studio.temperature")}
          icon={<HugeiconsIcon icon={TemperatureIcon} className="size-3.5" />}
          value={gpu.temperature_c != null ? `${gpu.temperature_c}°C` : "--"}
          pct={gpu.temperature_c ?? 0}
          max={100}
        />
        <GpuStat
          label={t("studio.vram")}
          icon={<HugeiconsIcon icon={RamMemoryIcon} className="size-3.5" />}
          value={gpu.vram_used_gb != null && gpu.vram_total_gb != null ? `${gpu.vram_used_gb} / ${gpu.vram_total_gb} GB` : "--"}
          pct={gpu.vram_utilization_pct ?? 0}
        />
        <GpuStat
          label={t("studio.power")}
          icon={<HugeiconsIcon icon={ZapIcon} className="size-3.5" />}
          value={gpu.power_draw_w != null ? gpu.power_limit_w != null ? `${gpu.power_draw_w} / ${gpu.power_limit_w} W` : `${gpu.power_draw_w} W` : "--"}
          pct={gpu.power_utilization_pct ?? 0}
        />
      </div>
    </div>
  );
};

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
  const { t } = useTranslation();
  
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
          <p className="text-xs font-semibold">{t("training.cfg").replace(/./g, "")}</p>
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
  const { t } = useTranslation();
  
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
          {stopRequested ? "Stopping..." : t("training.stopTraining")}
        </Button>
        <AlertDialogContent overlayClassName="bg-background/40 supports-backdrop-filter:backdrop-blur-[1px]">
          <AlertDialogHeader>
            <AlertDialogTitle>{t("training.stopTraining")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("training.stopChoice")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("training.continueTraining")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => onRequestStop(false)}
            >
              {t("training.cancelTraining")}
            </AlertDialogAction>
            <AlertDialogAction onClick={() => onRequestStop(true)}>
              {t("training.stopAndSave")}
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
  const { t } = useTranslation();
  
  if (!(showHalfwayHint || showCompletedHint)) {
    return null;
  }

  return (
    <div className="corner-squircle rounded-2xl border border-border/60 bg-muted/30 px-3 py-2.5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          {!showCompletedHint && (
            <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
              {t("training.milestone")}
            </p>
          )}
          <p
            className={cn(
              "text-xs text-foreground/85",
              !showCompletedHint && "mt-1",
            )}
          >
            {showCompletedHint
              ? t("training.doneNextStep")
              : t("training.halfwayDone")}
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
            {t("training.compareInChat")}
          </Button>
          <Button asChild={true} size="xs" variant="outline">
            <Link to="/export">{t("export.export")}</Link>
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
