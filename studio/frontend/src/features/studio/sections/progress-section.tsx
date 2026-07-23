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
import { getTrainingMethodLabel } from "@/features/training/lib/training-methods";
import type { TrainingViewData } from "@/features/training";
import type { RunConfigOverride } from "./run-config-override";
import { useGpuUtilization } from "@/hooks";
import type { GpuUtilization } from "@/hooks/use-gpu-utilization";
import { cn } from "@/lib/utils";
import {
  ChartAverageIcon,
  DashboardSpeed01Icon,
  FolderExportIcon,
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
} from "./progress-section-lib";
import { useT, type TranslationKey } from "@/i18n";

type ConfigGroup = {
  section: string;
  rows: [string, string | number | null | undefined][];
};

const phaseLabelKeys = {
  idle: "studio.progress.phase.idle",
  downloading_model: "studio.progress.phase.downloadingModel",
  downloading_dataset: "studio.progress.phase.downloadingDataset",
  loading_model: "studio.progress.phase.loadingModel",
  loading_dataset: "studio.progress.phase.loadingDataset",
  configuring: "studio.progress.phase.configuring",
  training: "studio.progress.phase.training",
  completed: "studio.progress.phase.completed",
  error: "studio.progress.phase.error",
  stopped: "studio.progress.phase.stopped",
} satisfies Record<TrainingViewData["phase"], TranslationKey>;

function configRow(
  label: string,
  value: string | number | null | undefined,
): [string, string | number | null | undefined] {
  return [label, value];
}

interface ProgressSectionProps {
  data: TrainingViewData;
  isHistorical?: boolean;
  configOverride?: RunConfigOverride;
}

export function ProgressSection({
  data,
  isHistorical = false,
  configOverride,
}: ProgressSectionProps): ReactElement {
  const t = useT();
  const navigate = useNavigate();
  const trainingMethodLabel = getTrainingMethodLabel(data.trainingMethod);

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

  // A finished run can be exported to GGUF: deep-link to the Export page with
  // this run preselected (its output-dir basename is the export model name).
  const exportRunName = data.outputDir
    ? (data.outputDir.replace(/[/\\]+$/, "").split(/[/\\]/).pop() || null)
    : null;
  const canExportGguf =
    !data.isTrainingRunning &&
    !!exportRunName &&
    !data.resumedLater &&
    (data.phase === "completed" || data.phase === "stopped");
  const handleExportGguf = () => {
    if (!exportRunName) return;
    void navigate({ to: "/export", search: { run: exportRunName } });
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

  // Prefer the run's saved snapshot when present (#6853). Live falls back to the
  // editable form store until it loads; History shows blanks, never live form values.
  const cfg = configOverride ?? (isHistorical ? undefined : config);
  const cfgEpochs = cfg?.epochs;
  const cfgBatchSize = cfg?.batchSize;
  const cfgLearningRate = cfg?.learningRate;
  const cfgMaxSteps = cfg?.maxSteps;
  const cfgContextLength = cfg?.contextLength;
  const cfgWarmupSteps = cfg?.warmupSteps;
  const cfgOptimizerType = cfg?.optimizerType;
  const cfgLoraRank = cfg?.loraRank;
  const cfgLoraAlpha = cfg?.loraAlpha;
  const cfgLoraDropout = cfg?.loraDropout;
  const cfgLoraVariant = cfg?.loraVariant;

  const optimizerLabel =
    OPTIMIZER_OPTIONS.find((o) => o.value === cfgOptimizerType)?.label ??
    cfgOptimizerType;

  const configItems: ConfigGroup[] = [
    {
      section: t("studio.progress.hyperparams"),
      rows: [
        configRow(t("studio.progress.epochs"), cfgEpochs),
        configRow(t("studio.progress.batchSize"), cfgBatchSize),
        configRow(t("studio.progress.learningRate"), cfgLearningRate),
        configRow(t("studio.progress.optimizer"), optimizerLabel),
        configRow(t("studio.progress.maxSteps"), cfgMaxSteps),
        configRow(t("studio.progress.contextLength"), cfgContextLength),
        configRow(t("studio.progress.warmupSteps"), cfgWarmupSteps),
      ],
    },
    ...(data.trainingMethod !== "full"
      ? [
        {
          section: "LoRA",
          rows: [
            configRow(t("studio.progress.rank"), cfgLoraRank),
            configRow(t("studio.progress.alpha"), cfgLoraAlpha),
            configRow(t("studio.progress.dropout"), cfgLoraDropout),
            configRow(t("studio.progress.variant"), cfgLoraVariant),
          ],
        },
      ]
      : []),
  ];

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
      title={t("studio.progress.title")}
      description={data.message || t("studio.progress.liveMetrics")}
      accent="emerald"
      className="shadow-border border border-border/60 bg-card/90 ring-0 backdrop-blur-sm"
      headerAction={
        <div className="flex items-center gap-2">
          {canExportGguf && (
            <Button
              size="sm"
              variant="outline"
              className="h-8 gap-1.5 text-xs"
              onClick={handleExportGguf}
            >
              <HugeiconsIcon icon={FolderExportIcon} className="size-3.5" />
              {t("studio.progress.exportGguf")}
            </Button>
          )}
          {isHistorical ? (
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
          )}
        </div>
      }
    >
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(18rem,0.8fr)]">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`rounded-full px-2.5 py-1 text-ui-10 font-semibold ${phaseColors[data.phase]}`}
            >
              {t(phaseLabelKeys[data.phase])}
            </span>
            {data.projectName && (
              <span className="rounded-full border border-border/60 px-2.5 py-1 text-ui-10 font-medium text-foreground/80">
                {data.projectName}
              </span>
            )}
            <span className="text-ui-10 tabular-nums text-muted-foreground">
              {t("studio.progress.epoch", {
                value: formatNumber(data.currentEpoch, 2),
              })}
            </span>
            <span className="rounded-full border border-border/60 px-2.5 py-1 text-ui-10 font-medium tabular-nums text-muted-foreground">
              {t("studio.progress.percentComplete", { percent: pct })}
            </span>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                {t("studio.progress.stepProgress", {
                  current: data.currentStep,
                  total: data.totalSteps || "--",
                })}
              </span>
              <span>{pct}%</span>
            </div>
            <Progress value={pct} className="h-2 bg-foreground/5" />
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

          <div
            className={cn(
              "grid gap-x-4 gap-y-3 pt-1 sm:grid-cols-2",
              data.projectName ? "xl:grid-cols-6" : "xl:grid-cols-5",
            )}
          >
            <MetricStat
              label={t("studio.progress.loss")}
              valueClassName="text-2xl font-bold tracking-tight"
            >
              {stoppedLoss != null ? stoppedLoss.toFixed(4) : "--"}
            </MetricStat>
            <MetricStat label={t("studio.progress.lr")}>{stoppedLr != null ? stoppedLr.toExponential(2) : "--"}</MetricStat>
            <MetricStat label={t("studio.progress.gradNorm")}>
              {formatNumber(stoppedGradNorm, 3)}
            </MetricStat>
            {data.projectName && (
              <MetricStat label={t("studio.progress.project")} valueClassName="truncate">
                {data.projectName}
              </MetricStat>
            )}
            <MetricStat label={t("studio.progress.model")} valueClassName="truncate">
              {data.modelName || "--"}
            </MetricStat>
            <MetricStat label={t("studio.progress.method")}>
              {trainingMethodLabel}
            </MetricStat>
          </div>

          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span>{t("studio.progress.elapsed", { value: formatDuration(elapsed) })}</span>
            {!isHistorical && (
              <span>{t("studio.progress.eta", { value: formatDuration(eta) })}</span>
            )}
            <span>
              {stepsPerSecond == null
                ? t("studio.progress.noStepsPerSecond")
                : t("studio.progress.stepsPerSecond", {
                  value: stepsPerSecond.toFixed(2),
                })}
            </span>
            {data.currentNumTokens != null && (
              <span>{t("studio.progress.tokens", { value: data.currentNumTokens })}</span>
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
  const t = useT();
  const [selectedGpu, setSelectedGpu] = useState(0);
  const gpuData = useGpuUtilization(isTrainingRunning);
  const gpus: GpuUtilization[] =
    Array.isArray(gpuData?.devices) && gpuData.devices.length > 0
      ? gpuData.devices
      : gpuData && Object.keys(gpuData).length > 0
        ? [gpuData]
        : [];

  useEffect(() => {
    if (selectedGpu > 0 && selectedGpu >= gpus.length) {
      setSelectedGpu(0);
    }
  }, [gpus.length, selectedGpu]);

  const gpuCount = gpus.length;
  const currentGpu: Partial<GpuUtilization> = gpus[selectedGpu] || gpus[0] || {};

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <p className="text-xs font-medium text-muted-foreground">
            {t("studio.progress.gpuMonitor")}
          </p>
          {gpuCount > 1 && (
            <select
              value={selectedGpu}
              onChange={(e) => setSelectedGpu(Number(e.target.value))}
              className="h-6 cursor-pointer rounded-md border border-border bg-popover px-1.5 py-0.5 text-ui-11 text-popover-foreground outline-none hover:bg-muted focus:border-ring transition-colors font-medium appearance-none"
              title="Select GPU"
            >
              {gpus.map((device, index) => (
                <option
                  key={device.index ?? index}
                  value={index}
                  className="bg-popover text-popover-foreground dark:bg-zinc-900 dark:text-zinc-100"
                >
                  GPU {device.visible_ordinal ?? index} - {device.backend} ({device.vram_total_gb ? `${Math.round(device.vram_total_gb)}GiB` : "N/A"})
                </option>
              ))}
            </select>
          )}
        </div>
        <span className="text-ui-11 text-muted-foreground">
          {t("studio.progress.live")}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2.5">
        <GpuStat
          label={t("studio.progress.utilization")}
          icon={<HugeiconsIcon icon={DashboardSpeed01Icon} className="size-3.5" />}
          value={
            currentGpu.gpu_utilization_pct != null
              ? `${currentGpu.gpu_utilization_pct}%`
              : "--"
          }
          pct={currentGpu.gpu_utilization_pct ?? 0}
        />
        <GpuStat
          label={t("studio.progress.temperature")}
          icon={<HugeiconsIcon icon={TemperatureIcon} className="size-3.5" />}
          value={
            currentGpu.temperature_c != null ? `${currentGpu.temperature_c}°C` : "--"
          }
          pct={currentGpu.temperature_c ?? 0}
          max={100}
        />
        <GpuStat
          label={t("studio.progress.vram")}
          icon={<HugeiconsIcon icon={RamMemoryIcon} className="size-3.5" />}
          value={
            currentGpu.vram_used_gb != null && currentGpu.vram_total_gb != null
              ? `${currentGpu.vram_used_gb} / ${currentGpu.vram_total_gb} GiB`
              : "--"
          }
          pct={currentGpu.vram_utilization_pct ?? 0}
        />
        <GpuStat
          label={t("studio.progress.power")}
          icon={<HugeiconsIcon icon={ZapIcon} className="size-3.5" />}
          value={
            currentGpu.power_draw_w != null
              ? currentGpu.power_limit_w != null
                ? `${currentGpu.power_draw_w} / ${currentGpu.power_limit_w} W`
                : `${currentGpu.power_draw_w} W`
              : "--"
          }
          pct={currentGpu.power_utilization_pct ?? 0}
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
  const t = useT();
  return (
    <Popover>
      <PopoverTrigger asChild={true}>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          className="rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
          aria-label={t("studio.progress.openConfig")}
        >
          <HugeiconsIcon icon={Notebook01Icon} className="size-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-72" align="end">
        <div className="flex flex-col gap-3">
          <p className="text-xs font-semibold">{t("studio.progress.configLabel")}</p>
          {configItems.map((group) => (
            <div key={group.section} className="flex flex-col gap-1">
              <p className="text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
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
  const t = useT();
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
          {stopRequested ? t("studio.training.stopping") : t("studio.training.stopAction")}
        </Button>
        <AlertDialogContent
          className="w-max max-w-[95vw]"
          overlayClassName="bg-background/40 supports-backdrop-filter:backdrop-blur-[1px]"
        >
          <AlertDialogHeader>
            <AlertDialogTitle>{t("studio.training.stopTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("studio.training.stopDescription")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("studio.training.continueAction")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => onRequestStop(false)}
            >
              {t("studio.training.cancelAction")}
            </AlertDialogAction>
            <AlertDialogAction onClick={() => onRequestStop(true)}>
              {t("studio.training.stopAndSave")}
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
  const t = useT();
  if (!(showHalfwayHint || showCompletedHint)) {
    return null;
  }

  return (
    <div className="corner-squircle rounded-2xl border border-border/60 bg-muted/30 px-3 py-2.5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          {!showCompletedHint && (
            <p className="text-ui-10 font-medium uppercase tracking-[0.12em] text-muted-foreground">
              {t("studio.training.milestone")}
            </p>
          )}
          <p
            className={cn(
              "text-xs text-foreground/85",
              !showCompletedHint && "mt-1",
            )}
          >
            {showCompletedHint
              ? t("studio.training.doneNextStep")
              : t("studio.training.halfwayDone")}
          </p>
        </div>
        {!showCompletedHint && (
          <span className="rounded-full border border-border/60 bg-background/80 px-2 py-0.5 text-ui-10 font-medium text-muted-foreground">
            50%+
          </span>
        )}
      </div>
      {showCompletedHint && (
        <div className="mt-2 flex flex-wrap gap-2">
          <Button size="xs" onClick={onCompareInChat}>
            {t("studio.training.compareInChat")}
          </Button>
          <Button asChild={true} size="xs" variant="outline">
            <Link to="/export">{t("studio.training.exportModel")}</Link>
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
      <p className="text-ui-11 text-muted-foreground">{label}</p>
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
    barColor = "bg-control-accent";
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
