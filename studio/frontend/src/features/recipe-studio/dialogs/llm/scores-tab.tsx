import { Button } from "@/components/ui/button";
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

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <FieldLabel
          label="Scorers"
          hint="Rubrics used by LLM Judge to score each generated row."
        />
        <Button type="button" size="xs" variant="outline" onClick={addScore}>
          Add scorer block
        </Button>
      </div>
      {scores.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Add scorer blocks. Each block spawns on graph and connects to this judge
          node.
        </p>
      )}
      {scores.map((score, index) => (
        <div
          key={`${config.id}-score-${index}`}
          className="flex items-center justify-between rounded-xl corner-squircle border border-border/60 px-3 py-2"
        >
          <div>
            <p className="text-xs font-semibold text-foreground">
              {score.name.trim() || `Scorer ${index + 1}`}
            </p>
            <p className="text-[11px] text-muted-foreground">
              {(score.options ?? []).length} options
            </p>
          </div>
          <Button
            type="button"
            size="xs"
            variant="ghost"
            onClick={() => removeScore(index)}
          >
            Remove
          </Button>
        </div>
      ))}
    </div>
  );
}
