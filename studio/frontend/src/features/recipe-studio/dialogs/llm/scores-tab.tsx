// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement } from "react";
import type { LlmConfig, Score } from "../../types";
import { FieldLabel } from "../shared/field-label";

type LlmScoresTabProps = {
  config: LlmConfig;
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

export function LlmScoresTab({
  config,
  onUpdate,
}: LlmScoresTabProps): ReactElement {
  const scores = config.scores ?? [];

  function updateScores(nextScores: Score[]): void {
    onUpdate({ scores: nextScores });
  }

  function removeScore(index: number): void {
    updateScores(scores.filter((_, currentIndex) => currentIndex !== index));
  }

  function addScore(): void {
    updateScores([
      ...scores,
      {
        name: "",
        description: "",
        options: [
          { value: "1", description: "" },
          { value: "5", description: "" },
        ],
      },
    ]);
  }

  function updateScore(index: number, patch: Partial<Score>): void {
    updateScores(
      scores.map((score, currentIndex) =>
        currentIndex === index ? { ...score, ...patch } : score,
      ),
    );
  }

  function addOption(scoreIndex: number): void {
    const score = scores[scoreIndex];
    if (!score) {
      return;
    }
    updateScore(scoreIndex, {
      options: [...(score.options ?? []), { value: "", description: "" }],
    });
  }

  function removeOption(scoreIndex: number, optionIndex: number): void {
    const score = scores[scoreIndex];
    if (!score) {
      return;
    }
    updateScore(scoreIndex, {
      options: (score.options ?? []).filter(
        (_option, currentIndex) => currentIndex !== optionIndex,
      ),
    });
  }

  function updateOption(
    scoreIndex: number,
    optionIndex: number,
    patch: { value?: string; description?: string },
  ): void {
    const score = scores[scoreIndex];
    if (!score) {
      return;
    }
    updateScore(scoreIndex, {
      options: (score.options ?? []).map((option, currentIndex) =>
        currentIndex === optionIndex ? { ...option, ...patch } : option,
      ),
    });
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <FieldLabel
          label="Scorers"
          hint="Rubrics used by LLM Judge to score each generated row."
        />
        {scores.length > 0 && (
          <Button type="button" size="xs" variant="outline" onClick={addScore}>
            Add scorer
          </Button>
        )}
      </div>
      {scores.length === 0 && (
        <Empty className="rounded-xl border border-dashed border-border/70 p-5">
          <EmptyHeader>
            <EmptyTitle className="text-sm">No scorers yet</EmptyTitle>
            <EmptyDescription className="text-xs">
              Add a scorer rubric before running judge generation.
            </EmptyDescription>
          </EmptyHeader>
          <EmptyContent className="max-w-none">
            <Button type="button" size="sm" onClick={addScore}>
              Add first scorer
            </Button>
          </EmptyContent>
        </Empty>
      )}
      {scores.map((score, index) => (
        <div
          key={`${config.id}-score-${index}`}
          className="space-y-2 rounded-xl corner-squircle border border-border/60 px-3 py-2"
        >
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-semibold text-foreground">
              {score.name.trim() || `Scorer ${index + 1}`}
            </p>
            <Button
              type="button"
              size="xs"
              variant="ghost"
              onClick={() => removeScore(index)}
            >
              Remove
            </Button>
          </div>
          <Input
            className="nodrag h-8 text-xs"
            placeholder="Score name"
            value={score.name}
            onChange={(event) =>
              updateScore(index, { name: event.target.value })
            }
          />
          <Textarea
            className="corner-squircle nodrag min-h-[56px] text-xs"
            placeholder="Score description"
            value={score.description}
            onChange={(event) =>
              updateScore(index, { description: event.target.value })
            }
          />
          <div className="space-y-1">
            {(score.options ?? []).map((option, optionIndex) => (
              <div
                key={`${config.id}-score-${index}-option-${optionIndex}`}
                className="grid grid-cols-[74px_1fr_auto] gap-1"
              >
                <Input
                  className="nodrag h-7 text-xs"
                  placeholder="Value"
                  value={option.value}
                  onChange={(event) =>
                    updateOption(index, optionIndex, {
                      value: event.target.value,
                    })
                  }
                />
                <Input
                  className="nodrag h-7 text-xs"
                  placeholder="Description"
                  value={option.description}
                  onChange={(event) =>
                    updateOption(index, optionIndex, {
                      description: event.target.value,
                    })
                  }
                />
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  onClick={() => removeOption(index, optionIndex)}
                >
                  x
                </Button>
              </div>
            ))}
            <Button
              type="button"
              size="xs"
              variant="outline"
              onClick={() => addOption(index)}
            >
              Add option
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}
