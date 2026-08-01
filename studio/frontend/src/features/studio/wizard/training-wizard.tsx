// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PICKER_TRIGGER_CLASS } from "@/components/resource-picker/picker-focus";
import { SegmentedTabsList } from "@/components/segmented-tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import { Tabs } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { HfTokenIndicator } from "@/features/hub";
import { TrainModelSelector } from "@/features/model-picker";
import { useSettingsDialogStore } from "@/features/settings";
import {
  TRAINING_METHOD_META,
  TRAINING_METHOD_ORDER,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { TrainingMethod } from "@/types/training";
import {
  BrainIcon,
  Database02Icon,
  FloppyDiskIcon,
  Settings05Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import type { CSSProperties, ReactNode } from "react";
import { DatasetPanel } from "../sections/dataset-section";
import { ParamsSection } from "../sections/params-section";
import { ConfigActions } from "./config-actions";
import type { ParamMode } from "./training-param-mode";

function SectionBox({
  title,
  description,
  icon,
  chipTint,
  titleAction,
  children,
  dataTour,
}: {
  title?: string;
  description?: string;
  icon?: IconSvgElement;
  chipTint?: string;
  titleAction?: ReactNode;
  children: ReactNode;
  dataTour?: string;
}) {
  return (
    <section
      data-tour={dataTour}
      className="@container/train-card elevated-card flex flex-col gap-4 bg-card p-5"
    >
      {title && (
        <div className="flex flex-col items-stretch gap-3 @md/train-card:flex-row @md/train-card:items-center @md/train-card:justify-between">
          <div className="flex min-w-0 items-center gap-3">
            {icon && (
              <span
                className="train-section-chip inline-flex size-9 shrink-0 items-center justify-center rounded-full"
                style={
                  chipTint
                    ? ({ "--chip-tint": chipTint } as CSSProperties)
                    : undefined
                }
              >
                <HugeiconsIcon
                  icon={icon}
                  strokeWidth={1.5}
                  className="size-[18px]"
                />
              </span>
            )}
            <div className="min-w-0">
              <h2 className="select-none text-ui-13p5 font-semibold leading-ui-18 tracking-[-0.012em] text-foreground">
                {title}
              </h2>
              {description && (
                <p className="text-ui-11p5 leading-ui-15 text-muted-foreground/85">
                  {description}
                </p>
              )}
            </div>
          </div>
          {titleAction && (
            <div className="w-full min-w-0 @md/train-card:w-auto @md/train-card:shrink-0">
              {titleAction}
            </div>
          )}
        </div>
      )}
      <div className="@container/train-section min-w-0">{children}</div>
    </section>
  );
}

function ParamModeToggle({
  mode,
  onChange,
}: {
  mode: ParamMode;
  onChange: (next: ParamMode) => void;
}) {
  const t = useT();
  const options = [
    { value: "simple", label: t("studio.params.mode.simple") },
    { value: "advanced", label: t("studio.params.mode.advanced") },
  ] as const;

  return (
    <Tabs
      value={mode}
      onValueChange={(value) => onChange(value as ParamMode)}
      className="contents"
    >
      <SegmentedTabsList
        value={mode}
        options={options}
        ariaLabel={t("studio.params.mode.ariaLabel")}
        size="compact"
        className="@md/train-card:w-[170px]"
      />
    </Tabs>
  );
}

function SetupField({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
        {label}
      </span>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

function TrainingMethodSelect() {
  const t = useT();
  const trainingMethod = useTrainingConfigStore((s) => s.trainingMethod);
  const setTrainingMethod = useTrainingConfigStore((s) => s.setTrainingMethod);
  const activeMeta = TRAINING_METHOD_META[trainingMethod];
  const activeLabel = activeMeta ? t(activeMeta.labelKey) : trainingMethod;
  return (
    <Select
      value={trainingMethod}
      onValueChange={(v) => setTrainingMethod(v as TrainingMethod)}
    >
      <SelectTrigger
        aria-label={`${t("studio.wizard.methodLabel")}: ${activeLabel}`}
        className={cn(
          PICKER_TRIGGER_CLASS,
          "w-full min-w-[148px] justify-between",
        )}
        data-tour="studio-method"
      >
        <span className="flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className={cn(
              "size-2 shrink-0 rounded-full",
              activeMeta?.dotClass ?? "bg-muted-foreground",
            )}
          />
          <span className="truncate font-medium text-foreground">
            {activeLabel}
          </span>
        </span>
      </SelectTrigger>
      <SelectContent
        position="popper"
        side="bottom"
        align="start"
        sideOffset={8}
        className="rounded-[14px] ring-0"
      >
        {TRAINING_METHOD_ORDER.map((method) => {
          const meta = TRAINING_METHOD_META[method];
          return (
            <Tooltip key={method} delayDuration={300}>
              <TooltipTrigger asChild={true} disableClickToggle={true}>
                <SelectItem value={method}>
                  <span className="flex items-center gap-2">
                    <span
                      aria-hidden="true"
                      className={cn(
                        "size-2 shrink-0 rounded-full",
                        meta.dotClass,
                      )}
                    />
                    {t(meta.labelKey)}
                  </span>
                </SelectItem>
              </TooltipTrigger>
              <TooltipContent
                side="right"
                sideOffset={10}
                className="max-w-[220px] text-ui-11p5 leading-snug"
              >
                {t(meta.hintKey)}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </SelectContent>
    </Select>
  );
}

function ModelPanel() {
  const t = useT();
  const openSettings = useSettingsDialogStore((s) => s.openDialog);
  return (
    <div className="grid grid-cols-1 gap-4 @md/train-section:grid-cols-2 @2xl/train-section:grid-cols-[minmax(0,1fr)_180px_200px]">
      <div className="@md/train-section:col-span-2 @2xl/train-section:col-span-1">
        <SetupField label={t("studio.wizard.modelLabel")}>
          <TrainModelSelector />
        </SetupField>
      </div>
      <SetupField label={t("studio.wizard.methodLabel")}>
        <TrainingMethodSelect />
      </SetupField>
      <div className="flex h-full min-w-0 items-end">
        <HfTokenIndicator
          showLabel={true}
          onOpenSettings={() => openSettings("general")}
        />
      </div>
    </div>
  );
}

export function TrainingWizard({
  paramMode,
  onParamModeChange,
}: {
  paramMode: ParamMode;
  onParamModeChange: (next: ParamMode) => void;
}) {
  const t = useT();
  return (
    <div className="flex flex-col gap-5">
      <SectionBox
        title={t("studio.wizard.modelTitle")}
        description={t("studio.wizard.modelDescription")}
        icon={BrainIcon}
        chipTint="var(--chart-1)"
        dataTour="studio-model"
      >
        <ModelPanel />
      </SectionBox>

      <SectionBox
        title={t("studio.wizard.datasetTitle")}
        description={t("studio.wizard.datasetDescription")}
        icon={Database02Icon}
        chipTint="var(--chart-4)"
        dataTour="studio-dataset"
      >
        <DatasetPanel />
      </SectionBox>

      <SectionBox
        title={t("studio.wizard.paramsTitle")}
        description={t("studio.wizard.paramsDescription")}
        icon={Settings05Icon}
        chipTint="var(--chart-5)"
        dataTour="studio-params"
        titleAction={
          <ParamModeToggle mode={paramMode} onChange={onParamModeChange} />
        }
      >
        <ParamsSection mode={paramMode} />
      </SectionBox>

      <SectionBox
        title={t("studio.wizard.configTitle")}
        description={t("studio.wizard.configDescription")}
        icon={FloppyDiskIcon}
        chipTint="var(--chart-3)"
      >
        <ConfigActions />
      </SectionBox>
    </div>
  );
}
