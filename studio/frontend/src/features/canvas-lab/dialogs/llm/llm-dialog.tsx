import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { ReactElement } from "react";
import type { LlmConfig } from "../../types";
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
  const modelAliasId = `${config.id}-model-alias`;
  const codeLangId = `${config.id}-code-lang`;
  const promptId = `${config.id}-prompt`;
  const outputFormatId = `${config.id}-output-format`;
  const systemPromptId = `${config.id}-system-prompt`;
  const updateField = <K extends keyof LlmConfig>(
    key: K,
    value: LlmConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<LlmConfig>);
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
        <Input
          id={modelAliasId}
          className="nodrag"
          value={config.model_alias}
          onChange={(event) => updateField("model_alias", event.target.value)}
        />
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
