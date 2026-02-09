import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, useEffect, useRef, useState } from "react";
import type { LlmConfig, Score } from "../../types";
import { AvailableVariables } from "./available-variables";
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
  modelConfigAliases: string[];
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

export function LlmDialog({
  config,
  modelConfigAliases,
  onUpdate,
}: LlmDialogProps): ReactElement {
  const modelAliasId = `${config.id}-model-alias`;
  const codeLangId = `${config.id}-code-lang`;
  const promptId = `${config.id}-prompt`;
  const outputFormatId = `${config.id}-output-format`;
  const systemPromptId = `${config.id}-system-prompt`;
  const modelAliasAnchorRef = useRef<HTMLDivElement>(null);
  const [aliasInput, setAliasInput] = useState(config.model_alias);
  useEffect(() => {
    setAliasInput(config.model_alias);
  }, [config.model_alias]);
  const scores = config.scores ?? [];
  const updateField = <K extends keyof LlmConfig>(
    key: K,
    value: LlmConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<LlmConfig>);
  };
  const updateScores = (next: Score[]) => updateField("scores", next);
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
  return (
    <div className="space-y-4">
      <AvailableVariables configId={config.id} />
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
            onInputValueChange={setAliasInput}
            itemToStringValue={(value) => value}
            autoHighlight={true}
          >
            <ComboboxInput
              id={modelAliasId}
              className="nodrag w-full"
              placeholder="Pick model alias or type"
              onBlur={() => {
                if (aliasInput !== config.model_alias) {
                  updateField("model_alias", aliasInput);
                }
              }}
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
          className="corner-squircle nodrag"
          value={config.prompt}
          onChange={(event) => updateField("prompt", event.target.value)}
        />
      </div>
      {config.llm_type === "judge" && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold uppercase text-muted-foreground">
              Scorers
            </p>
            <Button type="button" size="xs" variant="outline" onClick={addScore}>
              Add scorer block
            </Button>
          </div>
          {scores.length === 0 && (
            <p className="text-xs text-muted-foreground">
              Add scorer blocks. Each block spawns on canvas and connects to this judge node.
            </p>
          )}
          {scores.map((score, index) => (
            <div key={`${config.id}-score-${index}`} className="flex items-center justify-between rounded-xl corner-squircle border border-border/60 px-3 py-2">
              <div>
                <p className="text-xs font-semibold text-foreground">
                  {score.name.trim() || `Scorer ${index + 1}`}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  {(score.options ?? []).length} options
                </p>
              </div>
              <Button type="button" size="xs" variant="ghost" onClick={() => removeScore(index)}>
                Remove
              </Button>
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
            className="corner-squircle nodrag"
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
          className="corner-squircle nodrag"
          value={config.system_prompt}
          onChange={(event) => updateField("system_prompt", event.target.value)}
        />
      </div>
    </div>
  );
}
