import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, useMemo, useRef } from "react";
import { useCanvasLabStore } from "../../stores/canvas-lab";
import type { LlmConfig, Score, ScoreOption } from "../../types";
import { NameField } from "../shared/name-field";

const CODE_LANG_OPTIONS = [
  "python",
  "javascript",
  "typescript",
  "java",
  "kotlin",
  "go",
  "rust",
  "ruby",
  "scala",
  "swift",
  "sql:sqlite",
  "sql:postgres",
  "sql:mysql",
  "sql:tsql",
  "sql:bigquery",
  "sql:ansi",
];

type LlmDialogProps = {
  config: LlmConfig;
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

export function LlmDialog({ config, onUpdate }: LlmDialogProps): ReactElement {
  const configs = useCanvasLabStore((state) => state.configs);
  const modelConfigAliases = useMemo(
    () =>
      Object.values(configs)
        .filter((item) => item.kind === "model_config")
        .map((item) => item.name),
    [configs],
  );
  const modelAliasId = `${config.id}-model-alias`;
  const codeLangId = `${config.id}-code-lang`;
  const promptId = `${config.id}-prompt`;
  const outputFormatId = `${config.id}-output-format`;
  const systemPromptId = `${config.id}-system-prompt`;
  const modelAliasAnchorRef = useRef<HTMLDivElement>(null);
  const scores = config.scores ?? [];
  const updateField = <K extends keyof LlmConfig>(
    key: K,
    value: LlmConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<LlmConfig>);
  };
  const updateScores = (next: Score[]) => updateField("scores", next);
  const updateScore = (index: number, patch: Partial<Score>) => {
    updateScores(
      scores.map((score, i) =>
        i === index ? { ...score, ...patch } : score,
      ),
    );
  };
  const removeScore = (index: number) => {
    updateScores(scores.filter((_, i) => i !== index));
  };
  const addScore = () => {
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
  };
  const updateOption = (
    scoreIndex: number,
    optionIndex: number,
    patch: Partial<ScoreOption>,
  ) => {
    const score = scores[scoreIndex];
    if (!score) {
      return;
    }
    const nextOptions = score.options.map((option, i) =>
      i === optionIndex ? { ...option, ...patch } : option,
    );
    updateScore(scoreIndex, { options: nextOptions });
  };
  const addOption = (scoreIndex: number) => {
    const score = scores[scoreIndex];
    if (!score) {
      return;
    }
    updateScore(scoreIndex, {
      options: [...score.options, { value: "", description: "" }],
    });
  };
  const removeOption = (scoreIndex: number, optionIndex: number) => {
    const score = scores[scoreIndex];
    if (!score) {
      return;
    }
    updateScore(scoreIndex, {
      options: score.options.filter((_, i) => i !== optionIndex),
    });
  };
  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={modelAliasId}
        >
          Model alias
        </label>
        <div ref={modelAliasAnchorRef}>
          <Combobox
            items={modelConfigAliases}
            filteredItems={modelConfigAliases}
            filter={null}
            value={config.model_alias || null}
            onValueChange={(value) => updateField("model_alias", value ?? "")}
            onInputValueChange={(value) => updateField("model_alias", value)}
            itemToStringValue={(value) => value}
            autoHighlight={true}
          >
            <ComboboxInput
              id={modelAliasId}
              className="nodrag w-full"
              placeholder="Pick model alias or type"
            />
            <ComboboxContent anchor={modelAliasAnchorRef}>
              <ComboboxEmpty>No model configs found</ComboboxEmpty>
              <ComboboxList>
                {(alias: string) => (
                  <ComboboxItem key={alias} value={alias}>
                    {alias}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </div>
        <p className="text-xs text-muted-foreground">
          Pick a model config alias. Matching node link becomes semantic.
        </p>
      </div>
      {config.llm_type === "code" && (
        <div className="grid gap-2">
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor={codeLangId}
          >
            Code language
          </label>
          <Select
            value={config.code_lang ?? "python"}
            onValueChange={(value) => updateField("code_lang", value)}
          >
            <SelectTrigger className="nodrag w-full" id={codeLangId}>
              <SelectValue placeholder="Select language" />
            </SelectTrigger>
            <SelectContent>
              {CODE_LANG_OPTIONS.map((lang) => (
                <SelectItem key={lang} value={lang}>
                  {lang}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={promptId}
        >
          Prompt
        </label>
        <Textarea
          id={promptId}
          className="nodrag"
          value={config.prompt}
          onChange={(event) => updateField("prompt", event.target.value)}
        />
      </div>
      {config.llm_type === "judge" && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold uppercase text-muted-foreground">
              Scores
            </p>
            <Button type="button" size="xs" variant="outline" onClick={addScore}>
              Add score
            </Button>
          </div>
          {scores.length === 0 && (
            <p className="text-xs text-muted-foreground">
              Add at least one score to define evaluation criteria.
            </p>
          )}
          {scores.map((score, index) => (
            <div
              key={`${config.id}-score-${index}`}
              className="rounded-2xl border border-border/60 p-3"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="grid flex-1 gap-2">
                  <Input
                    className="nodrag"
                    placeholder="Score name (e.g., Relevance)"
                    value={score.name}
                    onChange={(event) =>
                      updateScore(index, { name: event.target.value })
                    }
                  />
                  <Textarea
                    className="nodrag"
                    placeholder="Score description and scoring guide"
                    value={score.description}
                    onChange={(event) =>
                      updateScore(index, { description: event.target.value })
                    }
                  />
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
              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold uppercase text-muted-foreground">
                    Options
                  </p>
                  <Button
                    type="button"
                    size="xs"
                    variant="outline"
                    onClick={() => addOption(index)}
                  >
                    Add option
                  </Button>
                </div>
                {score.options.map((option, optionIndex) => (
                  <div
                    key={`${config.id}-score-${index}-opt-${optionIndex}`}
                    className="flex items-start gap-2"
                  >
                    <Input
                      className="nodrag w-20"
                      placeholder="Value"
                      value={option.value}
                      onChange={(event) =>
                        updateOption(index, optionIndex, {
                          value: event.target.value,
                        })
                      }
                    />
                    <Textarea
                      className="nodrag min-h-[2.5rem] flex-1"
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
                      Remove
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      {config.llm_type === "structured" && (
        <div className="grid gap-2">
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor={outputFormatId}
          >
            Output format (JSON schema)
          </label>
          <Textarea
            id={outputFormatId}
            className="nodrag"
            value={config.output_format ?? ""}
            onChange={(event) =>
              updateField("output_format", event.target.value)
            }
          />
          <p className="text-xs text-muted-foreground">
            Paste a JSON schema object or minimal shape.
          </p>
        </div>
      )}
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={systemPromptId}
        >
          System prompt (optional)
        </label>
        <Textarea
          id={systemPromptId}
          className="nodrag"
          value={config.system_prompt}
          onChange={(event) => updateField("system_prompt", event.target.value)}
        />
      </div>
    </div>
  );
}
