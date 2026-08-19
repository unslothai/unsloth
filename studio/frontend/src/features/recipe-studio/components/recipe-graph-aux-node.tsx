// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Handle,
  Position,
  type Node,
  type NodeProps,
  useUpdateNodeInternals,
} from "@xyflow/react";
import { memo, type ReactElement, useEffect } from "react";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import type { LlmConfig, Score, ScoreOption } from "../types";
import { AUX_HANDLE_CLASS } from "../utils/handle-layout";
import { HANDLE_IDS } from "../utils/handles";
import { findInvalidJinjaReferences } from "../utils/refs";
import { getAvailableVariableEntries } from "../utils/variables";
import { AvailableReferencesInline } from "./shared/available-references-inline";
import { BaseNode, BaseNodeContent, BaseNodeHeader, BaseNodeHeaderTitle } from "./rf-ui/base-node";

type PromptField = "prompt" | "system_prompt";

type PromptInputNodeData = {
  kind: "llm-prompt-input";
  llmId: string;
  field: PromptField;
  title: string;
  executionLocked?: boolean;
};

type JudgeScoreNodeData = {
  kind: "llm-judge-score";
  llmId: string;
  scoreIndex: number;
  executionLocked?: boolean;
};

export type RecipeGraphAuxNodeData = PromptInputNodeData | JudgeScoreNodeData;
export type RecipeGraphAuxNodeType = Node<RecipeGraphAuxNodeData, "aux">;

function updateScoreAt(
  config: LlmConfig,
  scoreIndex: number,
  patch: Partial<Score>,
): Score[] {
  const scores = config.scores ?? [];
  return scores.map((score, index) =>
    index === scoreIndex ? { ...score, ...patch } : score,
  );
}

function updateOptionAt(
  score: Score,
  optionIndex: number,
  patch: Partial<ScoreOption>,
): ScoreOption[] {
  return score.options.map((option, index) =>
    index === optionIndex ? { ...option, ...patch } : option,
  );
}

function AuxVariableBadges({
  entries,
}: {
  entries: ReturnType<typeof getAvailableVariableEntries>;
}): ReactElement | null {
  return <AvailableReferencesInline entries={entries} />;
}

function AuxNodeBase({
  id,
  data,
}: NodeProps<RecipeGraphAuxNodeType>): ReactElement | null {
  const config = useRecipeStudioStore((state) => state.configs[data.llmId]);
  const configs = useRecipeStudioStore((state) => state.configs);
  const updateConfig = useRecipeStudioStore((state) => state.updateConfig);
  const updateNodeInternals = useUpdateNodeInternals();

  useEffect(() => {
    updateNodeInternals(id);
  }, [id, updateNodeInternals]);

  if (!(config && config.kind === "llm")) {
    return null;
  }
  const executionLocked = Boolean(data.executionLocked);

  const sourceHandles = (
    <>
      <Handle
        id={HANDLE_IDS.llmInputOutLeft}
        type="source"
        position={Position.Left}
        isConnectable={false}
        isConnectableStart={false}
        className={AUX_HANDLE_CLASS}
      />
      <Handle
        id={HANDLE_IDS.llmInputOutRight}
        type="source"
        position={Position.Right}
        isConnectable={false}
        isConnectableStart={false}
        className={AUX_HANDLE_CLASS}
      />
      <Handle
        id={HANDLE_IDS.llmInputOutTop}
        type="source"
        position={Position.Top}
        isConnectable={false}
        isConnectableStart={false}
        className={AUX_HANDLE_CLASS}
      />
      <Handle
        id={HANDLE_IDS.llmInputOutBottom}
        type="source"
        position={Position.Bottom}
        isConnectable={false}
        isConnectableStart={false}
        className={AUX_HANDLE_CLASS}
      />
    </>
  );

  if (data.kind === "llm-prompt-input") {
    const value = data.field === "prompt" ? config.prompt : config.system_prompt;
    const variableEntries = getAvailableVariableEntries(configs, data.llmId);
    const availableRefs = variableEntries.map((entry) => entry.name);
    const hasInvalidRefs =
      findInvalidJinjaReferences(value, availableRefs).length > 0;
    return (
      <BaseNode className="corner-squircle w-full min-w-0 rounded-4xl border-border/60 bg-card shadow-sm">
        <BaseNodeHeader className="border-b border-border/50 px-3 py-2">
          <BaseNodeHeaderTitle className="text-xs">{data.title}</BaseNodeHeaderTitle>
        </BaseNodeHeader>
        <BaseNodeContent className="gap-2 px-3 py-2">
          <Textarea
            className="corner-squircle nodrag nowheel max-h-40 min-h-[88px] w-full resize-none overflow-y-auto text-xs"
            aria-invalid={hasInvalidRefs}
            value={value}
            disabled={executionLocked}
            onChange={(event) =>
              updateConfig(data.llmId, {
                [data.field]: event.target.value,
              } as Partial<LlmConfig>)
            }
          />
          <AuxVariableBadges entries={variableEntries} />
        </BaseNodeContent>
        {sourceHandles}
      </BaseNode>
    );
  }

  const score = config.scores?.[data.scoreIndex];
  if (!score) {
    return null;
  }

  const updateScore = (patch: Partial<Score>): void => {
    updateConfig(data.llmId, {
      scores: updateScoreAt(config, data.scoreIndex, patch),
    });
  };

  const removeScore = (): void => {
    const nextScores = (config.scores ?? []).filter(
      (_score, index) => index !== data.scoreIndex,
    );
    updateConfig(data.llmId, { scores: nextScores });
  };

  const addOption = (): void => {
    updateScore({
      options: [...score.options, { value: "", description: "" }],
    });
  };

  const removeOption = (optionIndex: number): void => {
    updateScore({
      options: score.options.filter((_option, index) => index !== optionIndex),
    });
  };

  const updateOption = (
    optionIndex: number,
    patch: Partial<ScoreOption>,
  ): void => {
    updateScore({
      options: updateOptionAt(score, optionIndex, patch),
    });
  };

  return (
    <BaseNode className="corner-squircle w-full min-w-0 rounded-4xl border-border/60 bg-card shadow-sm">
      <BaseNodeHeader className="border-b border-border/50 px-3 py-2">
        <BaseNodeHeaderTitle className="text-xs">
          {score.name.trim() || `Scorer ${data.scoreIndex + 1}`}
        </BaseNodeHeaderTitle>
        <Button
          type="button"
          size="xs"
          variant="ghost"
          className="nodrag"
          disabled={executionLocked}
          onClick={removeScore}
        >
          Remove
        </Button>
      </BaseNodeHeader>
      <BaseNodeContent className="gap-2 px-3 py-2">
        <Input
          className="nodrag h-7 w-full text-xs"
          placeholder="Score name"
          value={score.name}
          disabled={executionLocked}
          onChange={(event) => updateScore({ name: event.target.value })}
        />
        <Textarea
          className="corner-squircle nodrag nowheel max-h-32 min-h-[56px] w-full resize-none overflow-y-auto text-xs"
          placeholder="Score description"
          value={score.description}
          disabled={executionLocked}
          onChange={(event) => updateScore({ description: event.target.value })}
        />
        <div className="space-y-1">
          {score.options.map((option, optionIndex) => (
            <div key={`${data.llmId}-score-${data.scoreIndex}-opt-${optionIndex}`} className="grid grid-cols-[74px_1fr_auto] gap-1">
              <Input
                className="nodrag h-7 text-xs"
                placeholder="Value"
                value={option.value}
                disabled={executionLocked}
                onChange={(event) =>
                  updateOption(optionIndex, { value: event.target.value })
                }
              />
              <Input
                className="nodrag h-7 text-xs"
                placeholder="Description"
                value={option.description}
                disabled={executionLocked}
                onChange={(event) =>
                  updateOption(optionIndex, {
                    description: event.target.value,
                  })
                }
              />
              <Button
                type="button"
                size="xs"
                variant="ghost"
                className="nodrag"
                disabled={executionLocked}
                onClick={() => removeOption(optionIndex)}
              >
                x
              </Button>
            </div>
          ))}
          <Button
            type="button"
            size="xs"
            variant="outline"
            className="nodrag mt-1"
            disabled={executionLocked}
            onClick={addOption}
          >
            Add option
          </Button>
        </div>
      </BaseNodeContent>
      {sourceHandles}
    </BaseNode>
  );
}

export const RecipeGraphAuxNode = memo(AuxNodeBase);
