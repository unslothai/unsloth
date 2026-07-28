// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { HfTokenIndicator } from "@/features/hub";
import {
  TRAIN_PICKER_TRIGGER_CLASS,
  TrainModelSelector,
} from "@/features/model-picker";
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
import {
  type CSSProperties,
  type ReactNode,
  useCallback,
  useState,
} from "react";
import { DatasetPanel } from "../sections/dataset-section";
import { ParamsSection } from "../sections/params-section";
import { ConfigActions } from "./config-actions";

function SectionBox({
  title,
  description,
  icon,
  iconColor,
  titleAction,
  children,
  dataTour,
}: {
  title?: string;
  description?: string;
  icon?: IconSvgElement;
  iconColor?: string;
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
                  iconColor
                    ? ({ "--chip-color": iconColor } as CSSProperties)
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
              <h3 className="select-none text-ui-13p5 font-semibold leading-ui-18 tracking-[-0.012em] text-foreground">
                {title}
              </h3>
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

type ParamMode = "simple" | "advanced";

const PARAM_MODE_KEY = "unsloth_train_param_mode";

function readParamMode(): ParamMode {
  if (typeof window === "undefined") {
    return "simple";
  }
  try {
    return window.localStorage.getItem(PARAM_MODE_KEY) === "advanced"
      ? "advanced"
      : "simple";
  } catch {
    return "simple";
  }
}

function useParamMode(): [ParamMode, (next: ParamMode) => void] {
  const [mode, setMode] = useState<ParamMode>(readParamMode);
  const update = useCallback((next: ParamMode) => {
    setMode(next);
    try {
      window.localStorage.setItem(PARAM_MODE_KEY, next);
    } catch {
      return;
    }
  }, []);
  return [mode, update];
}

function ParamModeToggle({
  mode,
  onChange,
}: {
  mode: ParamMode;
  onChange: (next: ParamMode) => void;
}) {
  const t = useT();
  return (
    <Tabs
      value={mode}
      onValueChange={(value) => onChange(value as ParamMode)}
      className="contents"
    >
      <TabsList
        unstyled={true}
        aria-label={t("studio.params.mode.ariaLabel")}
        className="hub-menu-trigger hub-tab-toggle relative inline-flex h-8 w-full shrink-0 items-center rounded-full @md/train-card:w-[170px]"
      >
        <span
          aria-hidden="true"
          className={cn(
            "hub-tab-toggle-pill pointer-events-none absolute inset-y-0 left-0 w-1/2 rounded-full transition-transform duration-200 ease-out",
            mode === "advanced" ? "translate-x-full" : "translate-x-0",
          )}
        />
        {(["simple", "advanced"] as const).map((value) => {
          const active = mode === value;
          return (
            <TabsTrigger
              key={value}
              value={value}
              indicatorClassName="hidden"
              className={cn(
                "relative z-10 h-8 flex-1 rounded-full border-0 px-3 py-0 text-ui-12p5",
                active
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {t(
                value === "simple"
                  ? "studio.params.mode.simple"
                  : "studio.params.mode.advanced",
              )}
            </TabsTrigger>
          );
        })}
      </TabsList>
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
  return (
    <Select
      value={trainingMethod}
      onValueChange={(v) => setTrainingMethod(v as TrainingMethod)}
    >
      <SelectTrigger
        className={cn(
          TRAIN_PICKER_TRIGGER_CLASS,
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
            {activeMeta ? t(activeMeta.labelKey) : trainingMethod}
          </span>
        </span>
      </SelectTrigger>
      <SelectContent
        position="popper"
        side="bottom"
        align="start"
        sideOffset={8}
        avoidCollisions={false}
        className="rounded-[14px] ring-0"
      >
        {TRAINING_METHOD_ORDER.map((method) => {
          const meta = TRAINING_METHOD_META[method];
          return (
            <Tooltip key={method} delayDuration={300}>
              <TooltipTrigger asChild={true}>
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
      <SetupField label={t("studio.wizard.hfTokenLabel")}>
        <HfTokenIndicator showLabel={true} />
      </SetupField>
    </div>
  );
}

export function TrainingWizard() {
  const t = useT();
  const [paramMode, setParamMode] = useParamMode();
  return (
    <div className="flex flex-col gap-5">
      <SectionBox
        title={t("studio.wizard.modelTitle")}
        description={t("studio.wizard.modelDescription")}
        icon={BrainIcon}
        iconColor="#7abf85"
        dataTour="studio-model"
      >
        <ModelPanel />
      </SectionBox>

      <SectionBox
        title={t("studio.wizard.datasetTitle")}
        description={t("studio.wizard.datasetDescription")}
        icon={Database02Icon}
        iconColor="#e7828c"
        dataTour="studio-dataset"
      >
        <DatasetPanel />
      </SectionBox>

      <SectionBox
        title={t("studio.wizard.paramsTitle")}
        description={t("studio.wizard.paramsDescription")}
        icon={Settings05Icon}
        iconColor="#8a7cce"
        dataTour="studio-params"
        titleAction={
          <ParamModeToggle mode={paramMode} onChange={setParamMode} />
        }
      >
        <ParamsSection mode={paramMode} />
      </SectionBox>

      <SectionBox
        title={t("studio.wizard.configTitle")}
        description={t("studio.wizard.configDescription")}
        icon={FloppyDiskIcon}
        iconColor="#6ab7be"
      >
        <ConfigActions />
      </SectionBox>
    </div>
  );
}
