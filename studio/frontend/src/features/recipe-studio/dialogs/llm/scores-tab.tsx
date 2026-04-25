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
import { useI18n } from "@/features/i18n";
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
  const { t } = useI18n();
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
          label={t("recipe.llm.scores.scorers")}
          hint={t("recipe.llm.scores.scorersHint")}
        />
        {scores.length > 0 && (
          <Button type="button" size="xs" variant="outline" onClick={addScore}>
            {t("recipe.llm.scores.addScorer")}
          </Button>
        )}
      </div>
      {scores.length === 0 && (
        <Empty className="rounded-xl border border-dashed border-border/70 p-5">
          <EmptyHeader>
            <EmptyTitle className="text-sm">{t("recipe.llm.scores.noneTitle")}</EmptyTitle>
            <EmptyDescription className="text-xs">
              {t("recipe.llm.scores.noneDescription")}
            </EmptyDescription>
          </EmptyHeader>
          <EmptyContent className="max-w-none">
            <Button type="button" size="sm" onClick={addScore}>
              {t("recipe.llm.scores.addFirstScorer")}
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
              {score.name.trim() || `${t("recipe.llm.scores.scorer")} ${index + 1}`}
            </p>
            <Button
              type="button"
              size="xs"
              variant="ghost"
              onClick={() => removeScore(index)}
            >
              {t("recipe.llm.scores.remove")}
            </Button>
          </div>
          <Input
            className="nodrag h-8 text-xs"
            placeholder={t("recipe.llm.scores.scoreName")}
            value={score.name}
            onChange={(event) =>
              updateScore(index, { name: event.target.value })
            }
          />
          <Textarea
            className="corner-squircle nodrag min-h-[56px] text-xs"
            placeholder={t("recipe.llm.scores.scoreDescription")}
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
                  placeholder={t("recipe.llm.scores.value")}
                  value={option.value}
                  onChange={(event) =>
                    updateOption(index, optionIndex, {
                      value: event.target.value,
                    })
                  }
                />
                <Input
                  className="nodrag h-7 text-xs"
                  placeholder={t("recipe.llm.scores.description")}
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
                  {t("recipe.llm.scores.remove")}
                </Button>
              </div>
            ))}
            <Button
              type="button"
              size="xs"
              variant="outline"
              onClick={() => addOption(index)}
            >
              {t("recipe.llm.scores.addOption")}
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}
