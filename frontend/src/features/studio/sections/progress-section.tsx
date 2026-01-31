import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useWizardStore } from "@/stores/training";
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
import type { ReactElement, ReactNode } from "react";

export function ProgressSection(): ReactElement | null {
  const store = useWizardStore();
  const metrics = store.trainingMetrics;
  if (!metrics) {
    return null;
  }

  const pct = Math.round((metrics.currentStep / metrics.totalSteps) * 100);
  const etaSec =
    metrics.totalSteps > 0
      ? Math.round(
          ((metrics.totalSteps - metrics.currentStep) /
            Math.max(metrics.currentStep, 1)) *
            metrics.elapsed,
        )
      : 0;
  const fmtTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}m ${sec}s`;
  };

  const statusColors = {
    training:
      "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300",
    warmup: "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
    saving:
      "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300",
  };
  const statusLabels = {
    training: "Training",
    warmup: "Warming up",
    saving: "Saving checkpoint",
  };

  const modelName = store.selectedModel ?? "—";

  const configItems = [
    {
      section: "Hyperparams",
      rows: [
        ["Epochs", store.epochs],
        ["Batch size", store.batchSize],
        ["Learning rate", store.learningRate],
        ["Max steps", store.maxSteps],
        ["Context length", store.contextLength],
        ["Warmup steps", store.warmupSteps],
      ],
    },
    ...(store.trainingMethod !== "full"
      ? [
          {
            section: "LoRA",
            rows: [
              ["Rank", store.loraRank],
              ["Alpha", store.loraAlpha],
              ["Dropout", store.loraDropout],
              ["Variant", store.loraVariant],
            ],
          },
        ]
      : []),
  ];

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
      title="Training Progress"
      description="Live training metrics"
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
          <Button
            variant="destructive"
            size="sm"
            className="h-7 cursor-pointer px-3 text-xs"
            onClick={() => store.setIsTraining(false)}
          >
            <HugeiconsIcon icon={StopIcon} className="size-3" /> Stop
          </Button>
        </div>
      }
    >
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {/* Left: Progress */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <span
              className={`rounded-full px-2.5 py-0.5 text-[10px] font-semibold ${statusColors[metrics.status]}`}
            >
              {statusLabels[metrics.status]}
            </span>
            <span className="text-[10px] tabular-nums text-muted-foreground">
              Epoch {metrics.currentEpoch.toFixed(2)} / {metrics.totalEpochs}
            </span>
          </div>

          {/* Progress bar */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                Step {metrics.currentStep} / {metrics.totalSteps}
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

          {/* Metrics */}
          <div className="flex items-baseline gap-4">
            <div>
              <p className="text-xs text-muted-foreground">Loss</p>
              <p className="text-3xl font-bold tabular-nums tracking-tight">
                {metrics.currentLoss.toFixed(4)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">LR</p>
              <p className="text-lg font-semibold tabular-nums">
                {metrics.currentLR.toExponential(2)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Grad Norm</p>
              <p className="text-lg font-semibold tabular-nums">
                {metrics.gradNorm.toFixed(3)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Model</p>
              <p className="text-lg font-semibold truncate max-w-[140px]">
                {modelName}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Method</p>
              <p className="text-lg font-semibold">{store.trainingMethod}</p>
            </div>
          </div>

          {/* Timings */}
          <div className="flex gap-4 text-xs text-muted-foreground">
            <span>Elapsed: {fmtTime(metrics.elapsed)}</span>
            <span>ETA: {fmtTime(etaSec)}</span>
            <span>{metrics.samplesPerSecond} samples/s</span>
          </div>
        </div>

        {/* Right: GPU */}
        <div className="flex flex-col gap-3">
          <p className="text-xs font-medium text-muted-foreground">
            GPU Monitor
          </p>
          <div className="grid grid-cols-2 gap-3">
            <GpuStat
              label="Utilization"
              icon={
                <HugeiconsIcon
                  icon={DashboardSpeed01Icon}
                  className="size-3.5"
                />
              }
              value={`${metrics.gpuUtil}%`}
              pct={metrics.gpuUtil}
            />
            <GpuStat
              label="Temperature"
              icon={
                <HugeiconsIcon icon={TemperatureIcon} className="size-3.5" />
              }
              value={`${metrics.gpuTemp}°C`}
              pct={metrics.gpuTemp}
              max={100}
            />
            <GpuStat
              label="VRAM"
              icon={<HugeiconsIcon icon={RamMemoryIcon} className="size-3.5" />}
              value={`${metrics.gpuVramUsed.toFixed(1)} / ${metrics.gpuVramTotal}GB`}
              pct={(metrics.gpuVramUsed / metrics.gpuVramTotal) * 100}
            />
            <GpuStat
              label="Power"
              icon={<HugeiconsIcon icon={ZapIcon} className="size-3.5" />}
              value={`${metrics.gpuPower}W`}
              pct={(metrics.gpuPower / 350) * 100}
            />
          </div>
        </div>
      </div>
    </SectionCard>
  );
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
