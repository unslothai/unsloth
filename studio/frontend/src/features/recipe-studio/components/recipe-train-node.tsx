// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  useTrainingActions,
  useTrainingRuntimeStore,
} from "@/features/training";
import { getTrainingMethodLabel } from "@/features/training/lib/training-methods";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Database02Icon,
  Rocket01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { NodeResizer, Position, useUpdateNodeInternals } from "@xyflow/react";
import { type ReactElement, useEffect, useMemo } from "react";
import { MAX_NODE_WIDTH, MIN_NODE_WIDTH } from "../constants";
import { useRecipeExecutionsStore } from "../stores/recipe-executions";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import type {
  RecipeNode as RecipeGraphNodeType,
  TrainingCardConfig,
} from "../types";
import { NODE_HANDLE_CLASS } from "../utils/handle-layout";
import { HANDLE_IDS } from "../utils/handles";
import {
  type WiredArtifact,
  applyTrainCardToTrainingStore,
  pickWiredArtifact,
  resolveTrainDataset,
} from "../utils/train-config";
import { RECIPE_STUDIO_NODE_TONES } from "../utils/ui-tones";
import {
  BaseNode,
  BaseNodeContent,
  BaseNodeHeader,
  BaseNodeHeaderTitle,
} from "./rf-ui/base-node";
import { LabeledHandle } from "./rf-ui/labeled-handle";

function formatLearningRate(value: number): string {
  if (!Number.isFinite(value) || value === 0) {
    return "0";
  }
  const exponent = value.toExponential(1);
  // 2.0e-4 -> 2e-4
  return exponent.replace(".0e", "e");
}

type StatChip = { label: string; value: string };

function buildStatChips(config: TrainingCardConfig): StatChip[] {
  return [
    { label: "Method", value: getTrainingMethodLabel(config.trainingMethod) },
    { label: "Epochs", value: String(config.epochs) },
    { label: "Rank", value: String(config.loraRank) },
    { label: "Batch", value: String(config.batchSize) },
    { label: "LR", value: formatLearningRate(config.learningRate) },
    { label: "Seq", value: String(config.contextLength) },
  ];
}

function describeDataset(
  config: TrainingCardConfig,
  wired: WiredArtifact | null,
): string {
  if (config.datasetSource === "huggingface") {
    return config.hfDataset.trim() || "Choose a Hugging Face dataset";
  }
  if (config.datasetSource === "upload") {
    return config.uploadedFile.trim() || "Choose a dataset file";
  }
  if (wired) {
    return `Auto-wired from ${wired.runLabel} run · ${wired.rows} rows`;
  }
  return "Waiting for a completed recipe run";
}

function describeDatasetSource(config: TrainingCardConfig): string {
  if (config.datasetSource === "huggingface") {
    return "Dataset source: Hugging Face (override)";
  }
  if (config.datasetSource === "upload") {
    return "Dataset source: uploaded file (override)";
  }
  return "Dataset source: upstream recipe output";
}

function resolveRunLabel(isRunning: boolean, isStarting: boolean): string {
  if (isRunning) {
    return "Training in progress";
  }
  return isStarting ? "Starting..." : "Run training";
}

type RecipeTrainNodeProps = {
  id: string;
  data: RecipeGraphNodeType["data"];
  selected: boolean;
};

export function RecipeTrainNode({
  id,
  data,
  selected,
}: RecipeTrainNodeProps): ReactElement | null {
  const config = useRecipeStudioStore((state) => state.configs[id]);
  const openConfig = useRecipeStudioStore((state) => state.openConfig);
  const executions = useRecipeExecutionsStore((state) => state.executions);
  const { startTrainingRun } = useTrainingActions();
  const isStarting = useTrainingRuntimeStore((state) => state.isStarting);
  const isTrainingRunning = useTrainingRuntimeStore(
    (state) => state.isTrainingRunning,
  );
  const startError = useTrainingRuntimeStore((state) => state.startError);
  const updateNodeInternals = useUpdateNodeInternals();
  const executionLocked = Boolean(data.executionLocked);

  // biome-ignore lint/correctness/useExhaustiveDependencies: re-measure handles when the graph orientation flips
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, data.layoutDirection, updateNodeInternals]);

  const wired = useMemo(() => pickWiredArtifact(executions), [executions]);

  if (config?.kind !== "train") {
    return null;
  }

  const dataset = resolveTrainDataset(config, wired);
  const hasDataset = Boolean(dataset.hfDataset || dataset.uploadedFile);
  const hasModel = Boolean(config.baseModel.trim());
  const busy = isStarting || isTrainingRunning;
  const canRun = !(executionLocked || busy) && hasDataset && hasModel;

  const stats = buildStatChips(config);
  const datasetLabel = describeDataset(config, wired);
  const datasetReady =
    config.datasetSource === "recipe" ? Boolean(wired) : hasDataset;
  const runLabel = resolveRunLabel(isTrainingRunning, isStarting);

  const onRun = () => {
    if (!canRun) {
      return;
    }
    applyTrainCardToTrainingStore(config, wired);
    startTrainingRun().catch(() => undefined);
  };

  return (
    <BaseNode
      className={cn(
        "corner-squircle relative w-full min-w-0 overflow-visible rounded-4xl border-border/60 shadow-sm",
        isTrainingRunning && "border-ring-strong shadow-md",
      )}
    >
      <NodeResizer
        isVisible={selected}
        minWidth={MIN_NODE_WIDTH}
        minHeight={160}
        maxWidth={MAX_NODE_WIDTH}
        maxHeight={640}
        color="var(--primary)"
        lineClassName="!border-transparent !shadow-none"
        lineStyle={{ opacity: 0 }}
        handleClassName="!h-3 !w-3 !border-transparent !bg-transparent"
        handleStyle={{ opacity: 0 }}
      />
      <BaseNodeHeader className="border-b border-border/50 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <div
            className={cn(
              "corner-squircle flex size-7 items-center justify-center rounded-md border",
              RECIPE_STUDIO_NODE_TONES.train,
            )}
          >
            <HugeiconsIcon icon={Rocket01Icon} className="size-3.5" />
          </div>
          <div className="min-w-0">
            <BaseNodeHeaderTitle className="truncate text-sm">
              {config.name}
            </BaseNodeHeaderTitle>
            <p className="truncate text-ui-11 text-muted-foreground">
              {data.subtype} · {config.baseModel.trim() || "No base model yet"}
            </p>
          </div>
        </div>
        <Button
          type="button"
          size="xs"
          variant="ghost"
          className="nodrag"
          disabled={executionLocked}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            openConfig(id);
          }}
        >
          Configure
        </Button>
      </BaseNodeHeader>

      <BaseNodeContent
        className={cn(
          "gap-3 px-3 py-3",
          executionLocked && "pointer-events-none opacity-85",
        )}
      >
        <div className="flex flex-wrap gap-1.5">
          {stats.map((stat) => (
            <Badge
              key={stat.label}
              variant="secondary"
              className="corner-squircle font-mono text-ui-11"
            >
              <span className="text-muted-foreground">{stat.label}</span>
              <span className="ml-1 text-foreground">{stat.value}</span>
            </Badge>
          ))}
        </div>

        <div
          className={cn(
            "corner-squircle flex items-center gap-2 rounded-2xl border px-3 py-2",
            datasetReady
              ? "border-emerald-500/40 bg-emerald-500/5"
              : "border-amber-400/50 bg-amber-500/5",
          )}
        >
          <HugeiconsIcon
            icon={datasetReady ? Tick02Icon : Database02Icon}
            className={cn(
              "size-4 shrink-0",
              datasetReady
                ? "text-emerald-600 dark:text-emerald-400"
                : "text-amber-600 dark:text-amber-400",
            )}
          />
          <div className="min-w-0">
            <p className="truncate text-xs font-medium text-foreground">
              {datasetLabel}
            </p>
            <p className="truncate text-ui-11 text-muted-foreground">
              {describeDatasetSource(config)}
            </p>
          </div>
        </div>

        {startError && (
          <div className="corner-squircle flex items-start gap-2 rounded-2xl border border-rose-500/40 bg-rose-500/5 px-3 py-2">
            <HugeiconsIcon
              icon={Alert02Icon}
              className="size-4 shrink-0 text-rose-600 dark:text-rose-400"
            />
            <p className="min-w-0 break-words text-ui-11 text-rose-700 dark:text-rose-300">
              {startError}
            </p>
          </div>
        )}

        <Button
          type="button"
          size="sm"
          className="nodrag w-full"
          disabled={!canRun}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            onRun();
          }}
        >
          {busy ? (
            <Spinner className="size-4" />
          ) : (
            <HugeiconsIcon icon={Rocket01Icon} className="size-4" />
          )}
          {runLabel}
        </Button>
        {!hasModel && (
          <p className="text-ui-11 text-muted-foreground">
            Pick a base model in settings to enable training.
          </p>
        )}
      </BaseNodeContent>

      <LabeledHandle
        id={HANDLE_IDS.dataIn}
        title="Data input"
        type="target"
        position={Position.Left}
        className="absolute inset-0 pointer-events-none"
        labelClassName="sr-only"
        handleClassName={NODE_HANDLE_CLASS}
      />
      <LabeledHandle
        id={HANDLE_IDS.dataInTop}
        title="Data input"
        type="target"
        position={Position.Top}
        className="absolute inset-0 pointer-events-none"
        labelClassName="sr-only"
        handleClassName={NODE_HANDLE_CLASS}
      />
      <LabeledHandle
        id={HANDLE_IDS.dataInRight}
        title="Data input"
        type="target"
        position={Position.Right}
        className="absolute inset-0 pointer-events-none opacity-0"
        labelClassName="sr-only"
        handleClassName={NODE_HANDLE_CLASS}
      />
      <LabeledHandle
        id={HANDLE_IDS.dataInBottom}
        title="Data input"
        type="target"
        position={Position.Bottom}
        className="absolute inset-0 pointer-events-none"
        labelClassName="sr-only"
        handleClassName={NODE_HANDLE_CLASS}
      />
    </BaseNode>
  );
}
