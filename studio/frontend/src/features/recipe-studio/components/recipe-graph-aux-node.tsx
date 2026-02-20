import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Handle,
  NodeResizer,
  Position,
  type Node,
  type NodeProps,
  useUpdateNodeInternals,
} from "@xyflow/react";
import { memo, type ReactElement, useEffect } from "react";
import { MAX_NODE_WIDTH, MIN_NODE_WIDTH } from "../constants";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import type { LayoutDirection, LlmConfig, Score, ScoreOption } from "../types";
import { HANDLE_IDS } from "../utils/handles";
import { getAvailableVariables } from "../utils/variables";
import { BaseNode, BaseNodeContent, BaseNodeHeader, BaseNodeHeaderTitle } from "./rf-ui/base-node";

type PromptField = "prompt" | "system_prompt";

type PromptInputNodeData = {
  kind: "llm-prompt-input";
  llmId: string;
  field: PromptField;
  title: string;
  layoutDirection: LayoutDirection;
};

type JudgeScoreNodeData = {
  kind: "llm-judge-score";
  llmId: string;
  scoreIndex: number;
  layoutDirection: LayoutDirection;
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

function AuxVariableBadges({ llmId }: { llmId: string }): ReactElement | null {
  const configs = useRecipeStudioStore((state) => state.configs);
  const vars = getAvailableVariables(configs, llmId);
  if (vars.length === 0) return null;
  return (
    <div className="space-y-1">
      <p className="text-[10px] font-medium text-muted-foreground">Available references</p>
      <div className="flex flex-wrap gap-1">
        {vars.map((v) => (
          <Badge
            key={v}
            variant="secondary"
            className="corner-squircle h-4 px-1.5 font-mono text-[10px]"
          >
            {v}
          </Badge>
        ))}
      </div>
    </div>
  );
}

function AuxNodeBase({
  id,
  data,
}: NodeProps<RecipeGraphAuxNodeType>): ReactElement | null {
  const config = useRecipeStudioStore((state) => state.configs[data.llmId]);
  const updateConfig = useRecipeStudioStore((state) => state.updateConfig);
  const updateNodeInternals = useUpdateNodeInternals();

  useEffect(() => {
    updateNodeInternals(id);
  }, [id, updateNodeInternals]);

  if (!(config && config.kind === "llm")) {
    return null;
  }

  const sourcePosition =
    data.layoutDirection === "TB" ? Position.Bottom : Position.Right;

  if (data.kind === "llm-prompt-input") {
    const value = data.field === "prompt" ? config.prompt : config.system_prompt;
    return (
      <BaseNode className="corner-squircle w-full min-w-0 rounded-lg border-border/60 bg-card shadow-sm">
        <NodeResizer
          isVisible={true}
          minWidth={MIN_NODE_WIDTH}
          minHeight={120}
          maxWidth={MAX_NODE_WIDTH}
          maxHeight={520}
          color="var(--primary)"
          lineClassName="!border-transparent !shadow-none"
          lineStyle={{ opacity: 0 }}
          handleClassName="!h-3 !w-3 !border-transparent !bg-transparent"
          handleStyle={{ opacity: 0 }}
        />
        <BaseNodeHeader className="border-b border-border/50 px-3 py-2">
          <BaseNodeHeaderTitle className="text-xs">{data.title}</BaseNodeHeaderTitle>
        </BaseNodeHeader>
        <BaseNodeContent className="gap-2 px-3 py-2">
          <Textarea
            className="corner-squircle nodrag max-h-40 min-h-[88px] w-full resize-none overflow-y-auto text-xs"
            value={value}
            onChange={(event) =>
              updateConfig(data.llmId, {
                [data.field]: event.target.value,
              } as Partial<LlmConfig>)
            }
          />
          <AuxVariableBadges llmId={data.llmId} />
        </BaseNodeContent>
        <Handle
          id={HANDLE_IDS.llmInputOut}
          type="source"
          position={sourcePosition}
          isConnectable={false}
          isConnectableStart={false}
          className="!size-2 !border-border !bg-background"
        />
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
    <BaseNode className="corner-squircle w-full min-w-0 rounded-lg border-border/60 bg-card shadow-sm">
      <NodeResizer
        isVisible={true}
        minWidth={MIN_NODE_WIDTH}
        minHeight={120}
        maxWidth={MAX_NODE_WIDTH}
        maxHeight={640}
        color="var(--primary)"
        lineClassName="!border-transparent !shadow-none"
        lineStyle={{ opacity: 0 }}
        handleClassName="!h-3 !w-3 !border-transparent !bg-transparent"
        handleStyle={{ opacity: 0 }}
      />
      <BaseNodeHeader className="border-b border-border/50 px-3 py-2">
        <BaseNodeHeaderTitle className="text-xs">
          {score.name.trim() || `Scorer ${data.scoreIndex + 1}`}
        </BaseNodeHeaderTitle>
        <Button type="button" size="xs" variant="ghost" className="nodrag" onClick={removeScore}>
          Remove
        </Button>
      </BaseNodeHeader>
      <BaseNodeContent className="gap-2 px-3 py-2">
        <Input
          className="nodrag h-7 w-full text-xs"
          placeholder="Score name"
          value={score.name}
          onChange={(event) => updateScore({ name: event.target.value })}
        />
        <Textarea
          className="corner-squircle nodrag max-h-32 min-h-[56px] w-full resize-none overflow-y-auto text-xs"
          placeholder="Score description"
          value={score.description}
          onChange={(event) => updateScore({ description: event.target.value })}
        />
        <div className="space-y-1">
          {score.options.map((option, optionIndex) => (
            <div key={`${data.llmId}-score-${data.scoreIndex}-opt-${optionIndex}`} className="grid grid-cols-[74px_1fr_auto] gap-1">
              <Input
                className="nodrag h-7 text-xs"
                placeholder="Value"
                value={option.value}
                onChange={(event) =>
                  updateOption(optionIndex, { value: event.target.value })
                }
              />
              <Input
                className="nodrag h-7 text-xs"
                placeholder="Description"
                value={option.description}
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
                onClick={() => removeOption(optionIndex)}
              >
                x
              </Button>
            </div>
          ))}
          <Button type="button" size="xs" variant="outline" className="nodrag mt-1" onClick={addOption}>
            Add option
          </Button>
        </div>
      </BaseNodeContent>
      <Handle
        id={HANDLE_IDS.llmInputOut}
        type="source"
        position={sourcePosition}
        isConnectable={false}
        isConnectableStart={false}
        className="!size-2 !border-border !bg-background"
      />
    </BaseNode>
  );
}

export const RecipeGraphAuxNode = memo(AuxNodeBase);
