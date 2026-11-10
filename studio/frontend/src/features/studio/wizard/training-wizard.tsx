// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { NewBadge } from "@/components/new-badge";
import { PICKER_TRIGGER_CLASS } from "@/components/resource-picker/picker-focus";
import { SectionBox, SectionField } from "@/components/section-box";
import { SegmentedTabsList } from "@/components/segmented-tabs";
import { Button } from "@/components/ui/button";
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
import {
  DECISION_MODEL_LABELS,
  useSettingsDialogStore,
} from "@/features/settings";
import { TrainModelSelector } from "@/features/train-model-picker";
import {
  type DecisionCheckpoint,
  TRAINING_METHOD_META,
  TRAINING_METHOD_ORDER,
  useTrainingConfigStore,
  effectiveTrainingObjective,
} from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import { ChipIcon, Settings01Icon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import type { TrainingMethod } from "@/types/training";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Award01Icon,
  BubbleChatEditIcon,
  Database02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useRef } from "react";
import { DatasetPanel } from "../sections/dataset-section";
import { DatasetSourceToggleAction } from "../sections/dataset-source-toggle";
import { FieldHint } from "../sections/field-hint";
import { ParamsSection } from "../sections/params-section";
import { ObjectiveSelect } from "../sections/rl/objective-select";
import { PromptFormatSection } from "../sections/rl/prompt-format-section";
import { RewardsSection } from "../sections/rl/rewards-section";
import { RlDatasetRoles } from "../sections/rl/rl-dataset-roles";
import { RlParamsSection } from "../sections/rl/rl-params-section";
import { ConfigActions } from "./config-actions";
import type { ParamMode } from "./training-param-mode";
import {
  type TrainingStep,
  resolveTrainingStep,
  trainingSteps,
} from "./training-steps";

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
        className="@md/train-card:w-[calc(180px*var(--ui-space-scale,1))]"
      />
    </Tabs>
  );
}

function SetupField({
  label,
  hint,
  badge,
  children,
}: {
  label: string;
  hint?: string;
  badge?: ReactNode;
  children: ReactNode;
}) {
  return (
    <SectionField
      label={label}
      hint={
        <>
          {hint ? <FieldHint text={hint} label={label} /> : null}
          {badge}
        </>
      }
    >
      {children}
    </SectionField>
  );
}

function TrainingMethodSelect() {
  const t = useT();
  const trainingMethod = useTrainingConfigStore((s) => s.trainingMethod);
  const setTrainingMethod = useTrainingConfigStore((s) => s.setTrainingMethod);
  const isDecision = useTrainingConfigStore((s) => s.modelType === "decision");
  // Clef and LLM decision models have an LLM backbone, so they take QLoRA; Laya does not.
  const decisionTakesQlora = useTrainingConfigStore(
    (s) =>
      s.modelType === "decision" &&
      (s.decisionLayout === "clef" || s.decisionLayout === "llm"),
  );
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
          "w-full min-w-[calc(148px*var(--ui-space-scale,1))] justify-between",
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
        className="rounded-xl ring-0"
      >
        {TRAINING_METHOD_ORDER.filter(
          (method) =>
            !isDecision ||
            method === "lora" ||
            method === "full" ||
            (decisionTakesQlora && method === "qlora"),
        ).map((method) => {
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
                className="max-w-[calc(220px*var(--ui-space-scale,1))] text-ui-11p5 leading-snug"
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

function TrainAsSelect() {
  const t = useT();
  const trainAsDecision = useTrainingConfigStore((s) => s.trainAsDecision);
  const setTrainAsDecision = useTrainingConfigStore(
    (s) => s.setTrainAsDecision,
  );
  const value = trainAsDecision ? "decision" : "language";
  const label = (option: string) =>
    option === "decision"
      ? t("studio.wizard.trainAsDecision")
      : t("studio.wizard.trainAsLanguage");
  return (
    <Select
      value={value}
      onValueChange={(next) => setTrainAsDecision(next === "decision")}
    >
      <SelectTrigger
        aria-label={`${t("studio.wizard.trainAsLabel")}: ${label(value)}`}
        className={cn(PICKER_TRIGGER_CLASS, "w-full justify-between")}
      >
        <span className="truncate font-medium text-foreground">
          {label(value)}
        </span>
      </SelectTrigger>
      <SelectContent
        position="popper"
        side="bottom"
        align="start"
        sideOffset={8}
        className="rounded-xl ring-0"
      >
        {["language", "decision"].map((option) => (
          <SelectItem key={option} value={option}>
            {label(option)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function CheckpointSelect({
  checkpoints,
}: {
  checkpoints: DecisionCheckpoint[];
}) {
  const t = useT();
  const modelSubfolder = useTrainingConfigStore((s) => s.modelSubfolder);
  const setModelSubfolder = useTrainingConfigStore((s) => s.setModelSubfolder);
  const active =
    checkpoints.find((checkpoint) => checkpoint.subfolder === modelSubfolder) ??
    checkpoints[0];
  const checkpointLabel = (checkpoint: DecisionCheckpoint) =>
    DECISION_MODEL_LABELS[checkpoint.name]
      ? t(DECISION_MODEL_LABELS[checkpoint.name])
      : checkpoint.name;
  return (
    <Select
      value={active.name}
      onValueChange={(name) => {
        const next = checkpoints.find((checkpoint) => checkpoint.name === name);
        if (next) {
          setModelSubfolder(next.subfolder);
        }
      }}
    >
      <SelectTrigger
        aria-label={`${t("studio.wizard.checkpointLabel")}: ${checkpointLabel(active)}`}
        className={cn(PICKER_TRIGGER_CLASS, "w-full justify-between")}
      >
        <span className="truncate font-medium text-foreground">
          {checkpointLabel(active)}
        </span>
      </SelectTrigger>
      <SelectContent
        position="popper"
        side="bottom"
        align="start"
        sideOffset={8}
        className="rounded-xl ring-0"
      >
        {checkpoints.map((checkpoint) => (
          <SelectItem key={checkpoint.name} value={checkpoint.name}>
            {checkpointLabel(checkpoint)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function ModelPanel() {
  const t = useT();
  const openSettings = useSettingsDialogStore((s) => s.openDialog);
  const decisionCheckpoints = useTrainingConfigStore((s) =>
    s.modelType === "decision" ? s.decisionCheckpoints : null,
  );
  const isDecisionModel = useTrainingConfigStore(
    (s) => s.modelType === "decision",
  );
  // Text and vision LLMs can also train as decision models; Laya, Clef, audio and embeddings cannot.
  const canTrainAsDecision = useTrainingConfigStore(
    (s) =>
      s.selectedModel !== null &&
      (s.decisionLayout === "llm" ||
        s.modelType === "text" ||
        s.modelType === "vision"),
  );
  return (
    <div className="grid grid-cols-1 gap-4 @md/train-section:grid-cols-2 @2xl/train-section:grid-cols-[minmax(0,1fr)_180px_200px]">
      <div className="@md/train-section:col-span-2 @2xl/train-section:col-span-1">
        <SetupField
          label={t("studio.wizard.modelLabel")}
          hint={t("studio.wizard.modelTooltip")}
        >
          <TrainModelSelector />
        </SetupField>
      </div>
      <SetupField
        label={t("studio.wizard.methodLabel")}
        hint={t("studio.wizard.methodTooltip")}
      >
        <TrainingMethodSelect />
      </SetupField>
      <SetupField
        label={t("picker.hfToken.label")}
        hint={t("studio.wizard.hfTokenDescription")}
      >
        <HfTokenIndicator
          showLabel={true}
          onOpenSettings={() => openSettings("general")}
        />
      </SetupField>
      {canTrainAsDecision ? (
        <SetupField
          label={t("studio.wizard.trainAsLabel")}
          hint={t("studio.wizard.trainAsTooltip")}
        >
          <TrainAsSelect />
        </SetupField>
      ) : null}
      {decisionCheckpoints && decisionCheckpoints.length > 0 ? (
        <SetupField
          label={t("studio.wizard.checkpointLabel")}
          hint={t("studio.wizard.checkpointTooltip")}
        >
          <CheckpointSelect checkpoints={decisionCheckpoints} />
        </SetupField>
      ) : null}
      {isDecisionModel ? null : (
        <div className="@md/train-section:col-span-2 @2xl/train-section:col-span-3">
          <SetupField
            label={t("rl.objective.label")}
            hint={t("rl.objective.tooltip")}
            badge={<NewBadge />}
          >
            <ObjectiveSelect />
          </SetupField>
        </div>
      )}
    </div>
  );
}

const STEP_TITLES = {
  model: "studio.wizard.modelTitle",
  dataset: "studio.wizard.datasetTitle",
  prompt: "rl.prompt.title",
  rewards: "rl.rewards.title",
  parameters: "studio.wizard.paramsTitle",
} as const satisfies Record<TrainingStep, TranslationKey>;

export function TrainingWizard({
  paramMode,
  onParamModeChange,
  step,
  onStepChange,
}: {
  paramMode: ParamMode;
  onParamModeChange: (next: ParamMode) => void;
  step: TrainingStep;
  onStepChange: (next: TrainingStep) => void;
}) {
  const t = useT();
  const objective = useTrainingConfigStore((s) =>
    effectiveTrainingObjective(s),
  );
  const steps = trainingSteps(objective);
  const active = resolveTrainingStep(step, objective);
  const index = steps.indexOf(active);
  const rootRef = useRef<HTMLDivElement>(null);

  const goTo = (next: TrainingStep) => {
    onStepChange(next);
    rootRef.current?.scrollIntoView({ block: "start" });
  };

  const panels: Record<TrainingStep, ReactNode> = {
    model: (
      <SectionBox
        title={t("studio.wizard.modelTitle")}
        description={t("studio.wizard.modelDescription")}
        icon={ChipIcon}
        chipTint="var(--chart-1)"
      >
        <ModelPanel />
      </SectionBox>
    ),
    dataset: (
      <SectionBox
        title={t("studio.wizard.datasetTitle")}
        description={t("studio.wizard.datasetDescription")}
        icon={Database02Icon}
        chipTint="var(--chart-1)"
        dataTour="studio-dataset"
        titleAction={<DatasetSourceToggleAction />}
      >
        <DatasetPanel />
        {objective !== "sft" && (
          <div className="mt-5">
            <RlDatasetRoles objective={objective} />
          </div>
        )}
      </SectionBox>
    ),
    prompt: (
      <SectionBox
        title={t("rl.prompt.title")}
        titleBadge={<NewBadge />}
        description={t("rl.prompt.description")}
        icon={BubbleChatEditIcon}
        chipTint="var(--chart-1)"
      >
        <PromptFormatSection />
      </SectionBox>
    ),
    rewards: (
      <SectionBox
        title={t("rl.rewards.title")}
        titleBadge={<NewBadge />}
        description={t("rl.rewards.description")}
        icon={Award01Icon}
        chipTint="var(--chart-1)"
      >
        <RewardsSection />
      </SectionBox>
    ),
    parameters: (
      <SectionBox
        title={t("studio.wizard.paramsTitle")}
        description={t("studio.wizard.paramsDescription")}
        icon={Settings01Icon}
        chipTint="var(--chart-1)"
        dataTour="studio-params"
        titleAction={
          <ParamModeToggle mode={paramMode} onChange={onParamModeChange} />
        }
      >
        <ParamsSection mode={paramMode} />
        {objective !== "sft" && paramMode === "simple" && (
          <div className="mt-5">
            <RlParamsSection objective={objective} />
          </div>
        )}
      </SectionBox>
    ),
  };

  return (
    <div ref={rootRef} className="flex scroll-mt-6 flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <nav aria-label={t("studio.wizard.stepsLabel")}>
          <ol className="flex flex-wrap items-center gap-1">
            {steps.map((item, position) => (
              <li key={item}>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  aria-current={item === active ? "step" : undefined}
                  className={cn(
                    "gap-2 px-2",
                    item === active
                      ? "bg-muted font-semibold text-foreground"
                      : "text-muted-foreground",
                  )}
                  onClick={() => goTo(item)}
                >
                  <span
                    aria-hidden="true"
                    className={cn(
                      "flex size-5 items-center justify-center rounded-full text-ui-11 tabular-nums",
                      item === active
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted",
                    )}
                  >
                    {position + 1}
                  </span>
                  {t(STEP_TITLES[item])}
                </Button>
              </li>
            ))}
          </ol>
        </nav>
        <ConfigActions />
      </div>

      {/* Hidden, not unmounted: drafts and test results survive moving between steps. */}
      {steps.map((item) => (
        <div key={item} hidden={item !== active} className="min-w-0">
          {panels[item]}
        </div>
      ))}

      <div className="flex items-center justify-between gap-3">
        <Button
          type="button"
          variant="outline"
          disabled={index === 0}
          onClick={() => goTo(steps[index - 1])}
        >
          <HugeiconsIcon
            icon={ArrowLeft01Icon}
            className="size-4 rtl:rotate-180"
          />
          {t("studio.wizard.back")}
        </Button>
        {index < steps.length - 1 && (
          <Button type="button" onClick={() => goTo(steps[index + 1])}>
            {t("studio.wizard.next", {
              step: t(STEP_TITLES[steps[index + 1]]),
            })}
            <HugeiconsIcon
              icon={ArrowRight01Icon}
              className="size-4 rtl:rotate-180"
            />
          </Button>
        )}
      </div>
    </div>
  );
}
