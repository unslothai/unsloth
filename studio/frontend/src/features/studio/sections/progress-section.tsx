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
import {
  useTrainingConfigStore,
  useTrainingActions,
  useTrainingRuntimeStore,
} from "@/features/training";
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
import { useState, type ReactElement, type ReactNode } from "react";
import { useShallow } from "zustand/react/shallow";
import { useGpuUtilization } from "@/hooks";
import { formatDuration, formatNumber, phaseColors, phaseLabel } from "./progress-section-lib";

export function ProgressSection(): ReactElement {
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
      loraRank: state.loraRank,
      loraAlpha: state.loraAlpha,
      loraDropout: state.loraDropout,
      loraVariant: state.loraVariant,
    })),
  );

  const { stopTrainingRun } = useTrainingActions();
  const gpu = useGpuUtilization(runtime.isTrainingRunning);
  const [stopDialogOpen, setStopDialogOpen] = useState(false);

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
    elapsed != null && elapsed > 0
      ? runtime.currentStep / elapsed
      : null;

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
    : lastNonZeroValue(runtime.gradNormHistory) ?? runtime.currentGradNorm;

  const configItems = [
    {
      section: "Hyperparams",
      rows: [
        ["Epochs", config.epochs],
        ["Batch size", config.batchSize],
        ["Learning rate", config.learningRate],
        ["Max steps", config.maxSteps],
        ["Context length", config.contextLength],
        ["Warmup steps", config.warmupSteps],
      ],
    },
    ...(config.trainingMethod !== "full"
      ? [
        {
          section: "LoRA",
          rows: [
            ["Rank", config.loraRank],
            ["Alpha", config.loraAlpha],
            ["Dropout", config.loraDropout],
            ["Variant", config.loraVariant],
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
      className="shadow-border ring-1 ring-border"
      headerAction={
        <div className="flex items-center gap-2">
          <Popover>
            <PopoverTrigger asChild={true}>
              <button
                type="button"
                className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors cursor-pointer"
              >
                <HugeiconsIcon icon={Notebook01Icon} className="size-4" />
              </button>
            </PopoverTrigger>
            <PopoverContent className="w-64" align="end">
              <div className="flex flex-col gap-3">
                <p className="text-xs font-semibold">Training Config</p>
                {configItems.map((group) => (
                  <div key={group.section} className="flex flex-col gap-1">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      {group.section}
                    </p>
                    {group.rows.map(([label, value]) => (
                      <div
                        key={String(label)}
                        className="flex justify-between text-xs"
                      >
                        <span className="text-muted-foreground">
                          {String(label)}
                        </span>
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
          <AlertDialog open={stopDialogOpen} onOpenChange={setStopDialogOpen}>
            <Button
              data-tour="studio-training-stop"
              variant="destructive"
              size="sm"
              className="h-7 cursor-pointer px-3 text-xs"
              onClick={() => setStopDialogOpen(true)}
              disabled={!runtime.isTrainingRunning}
            >
              <HugeiconsIcon icon={StopIcon} className="size-3" /> Stop
            </Button>
            <AlertDialogContent>
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
                  onClick={() => void stopTrainingRun(false)}
                >
                  Cancel Training
                </AlertDialogAction>
                <AlertDialogAction
                  onClick={() => void stopTrainingRun(true)}
                >
                  Stop and Save
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      }
    >
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <span
              className={`rounded-full px-2.5 py-0.5 text-[10px] font-semibold ${phaseColors[runtime.phase]}`}
            >
              {phaseLabel[runtime.phase]}
            </span>
            <span className="text-[10px] tabular-nums text-muted-foreground">
              Epoch {runtime.currentEpoch.toFixed(2)}
            </span>
          </div>

          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                Step {runtime.currentStep} / {runtime.totalSteps || "--"}
              </span>
              <span>{pct}%</span>
            </div>
            <div className="h-2.5 w-full rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-300"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>

          {runtime.error && (
            <p className="text-xs text-red-500 leading-relaxed">{runtime.error}</p>
          )}

          <div className="flex flex-wrap items-baseline gap-4">
            <div>
              <p className="text-xs text-muted-foreground">Loss</p>
              <p className="text-3xl font-bold tabular-nums tracking-tight">
                {stoppedLoss.toFixed(4)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">LR</p>
              <p className="text-lg font-semibold tabular-nums">
                {stoppedLr.toExponential(2)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Grad Norm</p>
              <p className="text-lg font-semibold tabular-nums">
                {formatNumber(stoppedGradNorm, 3)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Model</p>
              <p className="text-lg font-semibold truncate max-w-[140px]">
                {config.selectedModel ?? "--"}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Method</p>
              <p className="text-lg font-semibold">
                {config.trainingMethod.toUpperCase()}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
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
          <p className="text-xs font-medium text-muted-foreground">GPU Monitor</p>
          <div className="grid grid-cols-2 gap-3">
            <GpuStat
              label="Utilization"
              icon={
                <HugeiconsIcon
                  icon={DashboardSpeed01Icon}
                  className="size-3.5"
                />
              }
              value={gpu.gpu_utilization_pct != null ? `${gpu.gpu_utilization_pct}%` : "--"}
              pct={gpu.gpu_utilization_pct ?? 0}
            />
            <GpuStat
              label="Temperature"
              icon={<HugeiconsIcon icon={TemperatureIcon} className="size-3.5" />}
              value={gpu.temperature_c != null ? `${gpu.temperature_c}°C` : "--"}
              pct={gpu.temperature_c ?? 0}
              max={100}
            />
            <GpuStat
              label="VRAM"
              icon={<HugeiconsIcon icon={RamMemoryIcon} className="size-3.5" />}
              value={gpu.vram_used_gb != null && gpu.vram_total_gb != null ? `${gpu.vram_used_gb} / ${gpu.vram_total_gb} GB` : "--"}
              pct={gpu.vram_utilization_pct ?? 0}
            />
            <GpuStat
              label="Power"
              icon={<HugeiconsIcon icon={ZapIcon} className="size-3.5" />}
              value={gpu.power_draw_w != null ? (gpu.power_limit_w != null ? `${gpu.power_draw_w} / ${gpu.power_limit_w} W` : `${gpu.power_draw_w} W`) : "--"}
              pct={gpu.power_utilization_pct ?? 0}
            />
          </div>
        </div>
      </div>
    </SectionCard>
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
  pct,
  max,
}: {
  label: string;
  icon: ReactNode;
  value: string;
  pct: number;
  max?: number;
}): ReactElement {
  const clamped = Math.min(pct, max ?? 100);
  let barColor = "bg-red-500";
  if (clamped < 60) {
    barColor = "bg-emerald-500";
  } else if (clamped < 95) {
    barColor = "bg-amber-500";
  }

  return (
    <div className="flex flex-col gap-1.5 rounded-xl bg-muted/50 p-3">
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          {icon}
          {label}
        </span>
        <span className="font-medium tabular-nums">{value}</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full ${barColor} transition-all duration-300`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}
