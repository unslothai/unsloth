// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BalanceScaleIcon,
  Clock01Icon,
  CodeIcon,
  CodeSimpleIcon,
  DiceFaces03Icon,
  EqualSignIcon,
  FingerPrintIcon,
  FunctionIcon,
  Plug01Icon,
  Parabola02Icon,
  PencilEdit02Icon,
  Plant01Icon,
  Shield02Icon,
  Tag01Icon,
  TagsIcon,
  UserAccountIcon,
} from "@hugeicons/core-free-icons";
import { useMemo } from "react";
import type { Edge } from "@xyflow/react";
import { deriveDisplayGraph } from "../utils/graph/derive-display-graph";
import {
  deriveGraphRuntimeVisualState,
  pickLatestActiveExecution,
} from "../utils/graph/runtime-visual-state";
import type {
  LayoutDirection,
  LlmType,
  NodeConfig,
  RecipeNode as RecipeBuilderNode,
  SamplerType,
} from "../types";
import type { RecipeExecutionRecord } from "../execution-types";

type IconType = typeof CodeIcon;

const SAMPLER_ICONS: Record<SamplerType, IconType> = {
  category: Tag01Icon,
  subcategory: TagsIcon,
  uniform: EqualSignIcon,
  gaussian: Parabola02Icon,
  bernoulli: EqualSignIcon,
  datetime: Clock01Icon,
  timedelta: Clock01Icon,
  uuid: FingerPrintIcon,
  person: UserAccountIcon,
  person_from_faker: UserAccountIcon,
};

const LLM_ICONS: Record<LlmType, IconType> = {
  text: PencilEdit02Icon,
  structured: CodeIcon,
  code: CodeSimpleIcon,
  judge: BalanceScaleIcon,
};

function resolveExecutionColumnIcon(config: NodeConfig | null): IconType {
  if (!config) {
    return DiceFaces03Icon;
  }
  if (config.kind === "sampler") {
    return SAMPLER_ICONS[config.sampler_type];
  }
  if (config.kind === "llm") {
    return LLM_ICONS[config.llm_type];
  }
  if (config.kind === "expression") {
    return FunctionIcon;
  }
  if (config.kind === "validator") {
    return Shield02Icon;
  }
  if (config.kind === "seed") {
    return Plant01Icon;
  }
  if (config.kind === "model_provider") {
    return Shield02Icon;
  }
  if (config.kind === "model_config") {
    return Plant01Icon;
  }
  if (config.kind === "tool_config") {
    return Plug01Icon;
  }
  return PencilEdit02Icon;
}

type UseRecipeRuntimeVisualsArgs = {
  executions: RecipeExecutionRecord[];
  configs: Record<string, NodeConfig>;
  nodes: RecipeBuilderNode[];
  edges: Edge[];
  layoutDirection: LayoutDirection;
  auxNodePositions: Record<string, { x: number; y: number }>;
  llmAuxVisibility: Record<string, boolean>;
};

type UseRecipeRuntimeVisualsResult = {
  activeExecution: RecipeExecutionRecord | null;
  runtimeVisualState: ReturnType<typeof deriveGraphRuntimeVisualState>;
  displayGraph: ReturnType<typeof deriveDisplayGraph>;
  displayNodeIds: string[];
  currentColumnIcon: IconType;
};

export function useRecipeRuntimeVisuals({
  executions,
  configs,
  nodes,
  edges,
  layoutDirection,
  auxNodePositions,
  llmAuxVisibility,
}: UseRecipeRuntimeVisualsArgs): UseRecipeRuntimeVisualsResult {
  const activeExecution = useMemo(
    () => pickLatestActiveExecution(executions),
    [executions],
  );

  const runtimeVisualState = useMemo(
    () =>
      deriveGraphRuntimeVisualState({
        activeExecution,
        configs,
        edges,
      }),
    [activeExecution, configs, edges],
  );

  const displayGraph = useMemo(
    () =>
      deriveDisplayGraph({
        nodes,
        edges,
        configs,
        layoutDirection,
        auxNodePositions,
        llmAuxVisibility,
        runtime: runtimeVisualState,
      }),
    [
      auxNodePositions,
      configs,
      edges,
      layoutDirection,
      llmAuxVisibility,
      nodes,
      runtimeVisualState,
    ],
  );

  const currentColumnConfig = useMemo(() => {
    const columnName = activeExecution?.current_column?.trim();
    if (!columnName) {
      return null;
    }
    for (const config of Object.values(configs)) {
      if (config.name.trim() === columnName) {
        return config;
      }
    }
    return null;
  }, [activeExecution?.current_column, configs]);

  const currentColumnIcon = useMemo(
    () => resolveExecutionColumnIcon(currentColumnConfig),
    [currentColumnConfig],
  );

  const displayNodeIds = useMemo(
    () => displayGraph.nodes.map((node) => node.id),
    [displayGraph.nodes],
  );

  return {
    activeExecution,
    runtimeVisualState,
    displayGraph,
    displayNodeIds,
    currentColumnIcon,
  };
}
