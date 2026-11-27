// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionBox } from "@/components/section-box";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { useTrainingConfigStore } from "@/features/training";
import { useParallelPlan } from "@/features/training/hooks/use-parallel-plan";
import {
  PARALLEL_MODE_LABEL_KEYS,
  layoutExample,
  modeAvailability,
  overrideIssue,
  planSummary,
  showsGpuSection,
  toggleGpuSelection,
} from "@/features/training/lib/parallel-plan";
import { useTrainingParallelStore } from "@/features/training/stores/training-parallel-store";
import {
  PARALLEL_MODES,
  type ParallelGpu,
  type ParallelModeChoice,
} from "@/features/training/types/parallel";
import { isEngineGpuDevice, useGpuDevices } from "@/hooks/use-gpu-info";
import { useLocale, useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { GpuIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";

const MODE_CHOICES: ParallelModeChoice[] = ["auto", ...PARALLEL_MODES];

/** /api/system's torch CUDA / ROCm GPUs: gates the capabilities request, so one GPU asks nothing. */
function useSystemTrainingGpuCount(): number {
  const devices = useGpuDevices(true);
  return useMemo(() => devices.filter(isEngineGpuDevice).length, [devices]);
}

function formatGb(gb: number, locale: string): string {
  return new Intl.NumberFormat(locale, { maximumFractionDigits: 0 }).format(
    Math.round(gb),
  );
}

function DeviceRow({
  gpu,
  selectable,
  checked,
  onToggle,
}: {
  gpu: ParallelGpu;
  selectable: boolean;
  checked: boolean;
  onToggle: () => void;
}): ReactElement {
  const t = useT();
  const locale = useLocale();
  const label = t("studio.gpus.gpuIndex", { index: gpu.index });
  const body = (
    <>
      {selectable && (
        <Checkbox
          checked={checked}
          onCheckedChange={onToggle}
          aria-label={`${label}: ${gpu.name}`}
        />
      )}
      <span className="w-12 shrink-0 font-mono text-ui-11 text-muted-foreground/80">
        {label}
      </span>
      <span
        className="min-w-0 flex-1 truncate text-ui-12p5 text-foreground/90"
        title={gpu.name}
      >
        {gpu.name}
      </span>
      {gpu.totalGb !== null && (
        <span
          className="shrink-0 font-mono text-ui-11p5 text-muted-foreground"
          title={
            gpu.freeGb !== null
              ? t("studio.gpus.free", { gb: formatGb(gpu.freeGb, locale) })
              : undefined
          }
        >
          {t("studio.gpus.vram", { gb: formatGb(gpu.totalGb, locale) })}
        </span>
      )}
    </>
  );
  const className = cn(
    "flex min-h-9 items-center gap-3 rounded-lg px-3 py-1.5",
    "bg-[color-mix(in_oklab,var(--foreground)_calc(2.5%*var(--contrast-wash-gain,1)),transparent)] dark:bg-[rgb(255_255_255_/_calc(0.03*var(--contrast-wash-gain,1)))]",
    selectable && !checked && "opacity-60",
  );
  return selectable ? (
    // biome-ignore lint/a11y/noLabelWithoutControl: the Checkbox inside is the control.
    <label
      className={cn(className, "cursor-pointer")}
      data-gpu-index={gpu.index}
    >
      {body}
    </label>
  ) : (
    <div className={className} data-gpu-index={gpu.index}>
      {body}
    </div>
  );
}

function PlanLine({ hasModel }: { hasModel: boolean }): ReactElement | null {
  const t = useT();
  const locale = useLocale();
  const { status, plan, planError, hostGpuCount } = useTrainingParallelStore(
    useShallow((s) => ({
      status: s.planStatus,
      plan: s.plan,
      planError: s.planError,
      hostGpuCount: s.capabilities?.gpus.length ?? 1,
    })),
  );
  const count = useMemo(() => new Intl.NumberFormat(locale), [locale]);

  if (!hasModel) {
    return (
      <p className="text-ui-11p5 text-muted-foreground/85">
        {t("studio.gpus.pickModel")}
      </p>
    );
  }
  if (status === "unsupported") return null;
  if (status === "error") {
    return (
      <p className="text-ui-11p5 leading-relaxed text-amber-700 dark:text-amber-400">
        {t("studio.gpus.planFailed", { message: planError ?? "" })}
      </p>
    );
  }
  if (!plan) {
    return (
      <p className="flex items-center gap-2 text-ui-11p5 text-muted-foreground/85">
        <Spinner className="size-3" />
        {t("studio.gpus.planning")}
      </p>
    );
  }
  const summary = planSummary(plan, (n) => count.format(n), hostGpuCount);
  return (
    <div
      className={cn(
        "flex flex-col gap-2",
        status === "loading" && "opacity-60",
      )}
      data-testid="gpus-plan"
      data-mode={plan.mode}
    >
      {/* A refused choice resolves to some other mode; only the refusal is true for this choice. */}
      {!plan.error && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex h-5 items-center rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] px-2 text-ui-10p5 font-medium text-foreground/90 dark:bg-[rgb(255_255_255_/_calc(0.08*var(--contrast-wash-gain,1)))]">
            {t(PARALLEL_MODE_LABEL_KEYS[plan.mode])}
          </span>
          <p className="text-ui-12p5 text-foreground/90">
            {t(summary.key, summary.params)}
          </p>
        </div>
      )}
      {plan.reason && !plan.error && (
        <p className="text-ui-11 leading-relaxed text-muted-foreground/80">
          {plan.reason}
          {plan.perGpuGb !== null && (
            <span className="ml-1 font-mono">
              {t("studio.gpus.perGpu", { gb: plan.perGpuGb.toFixed(1) })}
            </span>
          )}
        </p>
      )}
      {plan.notes.map((note) => (
        <p
          key={note}
          className="text-ui-11 leading-relaxed text-muted-foreground/80"
        >
          {note}
        </p>
      ))}
      {plan.warnings.map((warning) => (
        <p
          key={warning}
          className="text-ui-11p5 leading-relaxed text-amber-700 dark:text-amber-400"
        >
          {warning}
        </p>
      ))}
      {plan.error && (
        <div
          className="flex flex-col gap-1 rounded-lg border border-destructive/30 bg-destructive/[0.06] px-3 py-2"
          role="alert"
          data-testid="gpus-plan-error"
          data-code={plan.error.code ?? undefined}
        >
          <p className="text-ui-11p5 leading-relaxed text-destructive">
            {plan.error.message}
          </p>
          {plan.error.alternative && (
            <p className="text-ui-11 leading-relaxed text-foreground/80">
              {t("studio.gpus.alternative", {
                alternative: plan.error.alternative,
              })}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function ModeSelect({
  allGpuIds,
}: { allGpuIds: readonly number[] }): ReactElement {
  const t = useT();
  const { override, plan, capabilities, setOverrideMode, setOverrideSpec } =
    useTrainingParallelStore(
      useShallow((s) => ({
        override: s.override,
        plan: s.plan,
        capabilities: s.capabilities,
        setOverrideMode: s.setOverrideMode,
        setOverrideSpec: s.setOverrideSpec,
      })),
    );
  const issue = overrideIssue(override, allGpuIds);
  const modeLabel = t(PARALLEL_MODE_LABEL_KEYS[override.mode]);
  const example = layoutExample((override.gpuIds ?? allGpuIds).length);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex min-h-9 items-center justify-between gap-3">
        <span className="text-xs font-medium text-muted-foreground">
          {t("studio.gpus.modeLabel")}
        </span>
        <Select
          value={override.mode}
          onValueChange={(value) =>
            setOverrideMode(value as ParallelModeChoice)
          }
        >
          <SelectTrigger
            className="w-[calc(240px*var(--ui-space-scale,1))] justify-between"
            aria-label={`${t("studio.gpus.modeLabel")}: ${modeLabel}`}
            data-testid="gpus-mode-trigger"
          >
            <span className="truncate">{modeLabel}</span>
          </SelectTrigger>
          <SelectContent
            position="popper"
            align="end"
            sideOffset={6}
            className="max-w-[calc(320px*var(--ui-space-scale,1))] rounded-xl ring-0"
          >
            {MODE_CHOICES.map((mode) => {
              const availability = modeAvailability(mode, plan, capabilities);
              const unavailable =
                availability !== null && !availability.available;
              return (
                <SelectItem key={mode} value={mode}>
                  <span className="flex flex-col">
                    <span
                      className={cn(unavailable && "text-muted-foreground")}
                    >
                      {t(PARALLEL_MODE_LABEL_KEYS[mode])}
                    </span>
                    {unavailable && (
                      <span className="whitespace-normal text-ui-10p5 text-muted-foreground">
                        {availability.reason ?? t("studio.gpus.unavailable")}
                      </span>
                    )}
                  </span>
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
      </div>
      {override.mode === "custom" && (
        <div className="flex min-h-9 items-center justify-between gap-3">
          <span className="text-xs font-medium text-muted-foreground">
            {t("studio.gpus.specLabel")}
          </span>
          <Input
            value={override.spec}
            onChange={(e) => setOverrideSpec(e.target.value)}
            placeholder={example}
            aria-label={t("studio.gpus.specLabel")}
            spellCheck={false}
            className="w-[calc(240px*var(--ui-space-scale,1))] font-mono"
          />
        </div>
      )}
      {issue && (
        <p
          className="text-ui-11p5 leading-relaxed text-destructive"
          role="alert"
        >
          {t(issue, { example })}
        </p>
      )}
    </div>
  );
}

function GpusPanel({ gpus }: { gpus: ParallelGpu[] }): ReactElement {
  const t = useT();
  const hasModel = useTrainingConfigStore((s) => Boolean(s.selectedModel));
  const { override, setOverrideEnabled, setOverrideGpuIds } =
    useTrainingParallelStore(
      useShallow((s) => ({
        override: s.override,
        setOverrideEnabled: s.setOverrideEnabled,
        setOverrideGpuIds: s.setOverrideGpuIds,
      })),
    );
  const allGpuIds = useMemo(() => gpus.map((g) => g.index), [gpus]);
  const selected = new Set(override.gpuIds ?? allGpuIds);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-1.5" data-testid="gpus-devices">
        {gpus.map((gpu) => (
          <DeviceRow
            key={gpu.index}
            gpu={gpu}
            selectable={override.enabled}
            checked={selected.has(gpu.index)}
            onToggle={() =>
              setOverrideGpuIds(
                toggleGpuSelection(override.gpuIds, gpu.index, allGpuIds),
              )
            }
          />
        ))}
      </div>
      <PlanLine hasModel={hasModel} />
      <Collapsible open={override.enabled} onOpenChange={setOverrideEnabled}>
        <CollapsibleTrigger
          className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground"
          data-testid="gpus-override-trigger"
        >
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            className={`size-3.5 transition-transform ${override.enabled ? "rotate-180" : ""}`}
          />
          {t("studio.gpus.override")}
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-4 flex flex-col gap-2 data-[state=open]:overflow-visible">
          <p className="text-ui-11 leading-relaxed text-muted-foreground/85">
            {t("studio.gpus.overrideHint")}
          </p>
          <ModeSelect allGpuIds={allGpuIds} />
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

/** Hidden on a 0 or 1 GPU host and on a backend without the multi-GPU routes. */
export function GpusSection(): ReactElement | null {
  const t = useT();
  const systemGpuCount = useSystemTrainingGpuCount();
  useParallelPlan(systemGpuCount);
  const capabilities = useTrainingParallelStore((s) => s.capabilities);
  if (
    !showsGpuSection(systemGpuCount) ||
    !capabilities ||
    !showsGpuSection(capabilities.gpus.length)
  ) {
    return null;
  }
  return (
    <SectionBox
      title={t("studio.gpus.title")}
      description={t("studio.gpus.description")}
      icon={GpuIcon}
      chipTint="var(--chart-1)"
    >
      <GpusPanel gpus={capabilities.gpus} />
    </SectionBox>
  );
}
