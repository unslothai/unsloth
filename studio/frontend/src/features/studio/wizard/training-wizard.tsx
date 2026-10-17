// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { NewBadge } from "@/components/new-badge";
import { Button } from "@/components/ui/button";
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
import { usePlatformStore } from "@/config/env";
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
  isTrainingMethodSupportedOnDevice,
  useTrainingConfigStore,
  effectiveTrainingObjective,
} from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { TrainingMethod } from "@/types/training";
import {
  AiBrain01Icon,
  BubbleChatEditIcon,
  Database02Icon,
  Settings05Icon,
} from "@hugeicons/core-free-icons";
import { ArrowLeftIcon, ArrowRightIcon } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { DatasetPanel } from "../sections/dataset-section";
import { DatasetSourceToggleAction } from "../sections/dataset-source-toggle";
import { FieldHint } from "../sections/field-hint";
import { SectionBox } from "../sections/section-box";
import { ParamsSection } from "../sections/params-section";
import { ObjectiveSelect } from "../sections/rl/objective-select";
import { PromptFormatSection } from "../sections/rl/prompt-format-section";
import { RewardsSection } from "../sections/rl/rewards-section";
import { RlDatasetRoles } from "../sections/rl/rl-dataset-roles";
import { RlParamsSection } from "../sections/rl/rl-params-section";
import { ConfigActions } from "./config-actions";
import type { ParamMode } from "./training-param-mode";
import { RunPreviewCard } from "./run-preview-card";
import { StartTrainingCta } from "./start-training-cta";
import {
  resolveTrainingStep,
  trainingSteps,
  type TrainingStep,
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
    <div className="flex flex-col gap-2">
      <span className="flex items-center gap-1.5 text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
        {label}
        {hint ? <FieldHint text={hint} label={label} /> : null}
        {badge}
      </span>
      <div className="min-w-0">{children}</div>
    </div>
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
  const deviceType = usePlatformStore((state) => state.deviceType);
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
        className="rounded-[14px] ring-0"
      >
        {TRAINING_METHOD_ORDER.filter(
          (method) =>
            !isDecision ||
            method === "lora" ||
            method === "full" ||
            (decisionTakesQlora && method === "qlora"),
        ).map((method) => {
          const meta = TRAINING_METHOD_META[method];
          const unsupportedOnMlx = !isTrainingMethodSupportedOnDevice(
            method,
            deviceType,
          );
          return (
            <Tooltip key={method} delayDuration={300}>
              <TooltipTrigger asChild={true} disableClickToggle={true}>
                <SelectItem value={method} disabled={unsupportedOnMlx}>
                  <span className="flex items-center gap-2">
                    <span
                      aria-hidden="true"
                      className={cn(
                        "size-2 shrink-0 rounded-full",
                        meta.dotClass,
                      )}
                    />
                    {t(meta.labelKey)}
                    {unsupportedOnMlx && (
                      <span className="text-ui-10 text-muted-foreground">
                        · {t("studio.params.notSupportedAppleSilicon")}
                      </span>
                    )}
                  </span>
                </SelectItem>
              </TooltipTrigger>
              <TooltipContent
                side="right"
                sideOffset={10}
                className="max-w-[calc(220px*var(--ui-space-scale,1))] text-ui-11p5 leading-snug"
              >
                {unsupportedOnMlx
                  ? t("studio.params.notSupportedAppleSilicon")
                  : t(meta.hintKey)}
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
        className="rounded-[14px] ring-0"
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
        className="rounded-[14px] ring-0"
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

const STEP_LABELS = {
  model: "studio.wizard.modelTitle",
  dataset: "studio.wizard.datasetTitle",
  prompt: "studio.wizard.promptStep",
  rewards: "rl.rewards.title",
  parameters: "studio.wizard.settingsStep",
  review: "studio.wizard.review",
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
  const objective = useTrainingConfigStore((s) => effectiveTrainingObjective(s));
  const steps = trainingSteps(objective);
  const activeStep = resolveTrainingStep(step, objective);
  const index = steps.indexOf(activeStep);
  const reducedMotion = useReducedMotion();
  const [direction, setDirection] = useState(1);
  const containerRef = useRef<HTMLDivElement>(null);
  const panelRefs = useRef<Partial<Record<TrainingStep, HTMLDivElement>>>({});
  const focusOnChange = useRef(false);

  useEffect(() => {
    if (activeStep !== step) onStepChange(activeStep);
    if (!focusOnChange.current) return;
    focusOnChange.current = false;
    panelRefs.current[activeStep]?.focus({ preventScroll: true });
    containerRef.current?.scrollIntoView({ block: "start", behavior: "instant" });
  }, [activeStep, step, onStepChange]);

  function goTo(next: TrainingStep) {
    if (next === activeStep) return;
    setDirection(steps.indexOf(next) > index ? 1 : -1);
    focusOnChange.current = true;
    onStepChange(next);
  }

  const panels: Record<TrainingStep, ReactNode> = {
    model: (
      <SectionBox
        title={t("studio.wizard.modelTitle")}
        description={t("studio.wizard.modelDescription")}
        icon={AiBrain01Icon}
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
        chipTint="var(--chart-4)"
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
    rewards: <RewardsSection />,
    parameters: (
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
        {objective !== "sft" && paramMode === "simple" && (
          <div className="mt-5">
            <RlParamsSection objective={objective} />
          </div>
        )}
      </SectionBox>
    ),
    review: <RunPreviewCard paramMode={paramMode} />,
  };

  return (
    <div
      ref={containerRef}
      className="@container/training-wizard flex min-w-0 scroll-mt-6 flex-col gap-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p
          aria-live="polite"
          aria-atomic="true"
          className="text-sm tabular-nums text-muted-foreground"
        >
          {t("studio.wizard.stepProgress", {
            current: index + 1,
            total: steps.length,
          })}
        </p>
        <ConfigActions />
      </div>
      <nav aria-label={t("studio.wizard.setupNavigation")}>
        <ol className={cn(
          "grid gap-1",
          steps.length === 6
            ? "grid-cols-3 @3xl/training-wizard:grid-cols-6"
            : "grid-cols-2 @xl/training-wizard:grid-cols-4",
        )}>
          {steps.map((item, position) => (
            <li key={item} className="min-w-0">
              <Button
                id={`training-step-${item}`}
                type="button"
                variant="ghost"
                className={cn(
                  "h-auto min-h-10 w-full justify-start gap-2 whitespace-nowrap px-2 py-2 text-start text-ui-12",
                  item === activeStep
                    ? "bg-muted text-foreground"
                    : "text-muted-foreground",
                )}
                aria-current={item === activeStep ? "step" : undefined}
                aria-controls={`training-panel-${item}`}
                onClick={() => goTo(item)}
              >
                <span
                  aria-hidden="true"
                  className={cn(
                    "flex size-6 shrink-0 items-center justify-center rounded-full text-xs tabular-nums",
                    item === activeStep
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground",
                  )}
                >
                  {position + 1}
                </span>
                <span className="min-w-0 truncate">
                  {t(STEP_LABELS[item])}
                </span>
              </Button>
            </li>
          ))}
        </ol>
      </nav>

      {/* Keep panels mounted so back/forward retains local drafts and test results. */}
      {steps.map((item) => (
        <motion.div
          key={item}
          id={`training-panel-${item}`}
          ref={(element) => {
            panelRefs.current[item] = element ?? undefined;
          }}
          role="region"
          aria-labelledby={`training-step-${item}`}
          tabIndex={-1}
          hidden={item !== activeStep}
          className="min-w-0 outline-none"
          initial={false}
          animate={{
            opacity: item === activeStep ? 1 : 0.6,
            x: reducedMotion || item === activeStep ? 0 : direction * 12,
          }}
          transition={{
            duration: reducedMotion ? 0 : 0.18,
            ease: [0.16, 1, 0.3, 1],
          }}
        >
          {panels[item]}
        </motion.div>
      ))}

      <div className="sticky bottom-0 z-10 flex items-start justify-between gap-3 border-t border-border/60 bg-background py-4">
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <span className="inline-flex shrink-0">
              <Button
                type="button"
                variant="outline"
                size="icon-lg"
                className="size-11 rounded-full"
                aria-label={t("studio.wizard.back")}
                disabled={index === 0}
                onClick={() => goTo(steps[index - 1])}
              >
                <ArrowLeftIcon className="size-5 rtl:rotate-180" strokeWidth={1.75} />
              </Button>
            </span>
          </TooltipTrigger>
          <TooltipContent>{t("studio.wizard.back")}</TooltipContent>
        </Tooltip>
        {activeStep === "review" ? (
          <div className="min-w-0 flex-1">
            <StartTrainingCta />
          </div>
        ) : (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <Button
                type="button"
                size="icon-lg"
                className="size-11 rounded-full"
                aria-label={t("studio.wizard.continue")}
                onClick={() => goTo(steps[index + 1])}
              >
                <ArrowRightIcon className="size-5 rtl:rotate-180" strokeWidth={1.75} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t("studio.wizard.continue")}</TooltipContent>
          </Tooltip>
        )}
      </div>
    </div>
  );
}
