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
import { type ReactElement, type RefObject } from "react";
import type { LlmConfig } from "../../types";
import { AvailableVariables } from "../shared/available-variables";
import { FieldLabel } from "../shared/field-label";
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

type LlmGeneralTabProps = {
  config: LlmConfig;
  modelConfigAliases: string[];
  modelProviderOptions: string[];
  modelAliasAnchorRef: RefObject<HTMLDivElement | null>;
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

export function LlmGeneralTab({
  config,
  modelConfigAliases,
  modelProviderOptions,
  modelAliasAnchorRef,
  onUpdate,
}: LlmGeneralTabProps): ReactElement {
  const modelAliasId = `${config.id}-model-alias`;
  const codeLangId = `${config.id}-code-lang`;
  const promptId = `${config.id}-prompt`;
  const outputFormatId = `${config.id}-output-format`;
  const systemPromptId = `${config.id}-system-prompt`;
  const hasModelConfigs = modelConfigAliases.length > 0;
  const hasModelProviders = modelProviderOptions.length > 0;

  return (
    <div className="space-y-4">
      <AvailableVariables configId={config.id} />
      <NameField value={config.name} onChange={(value) => onUpdate({ name: value })} />
      {(!hasModelConfigs || !hasModelProviders) && (
        <div className="rounded-2xl border border-border/60 bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
          <p className="font-semibold text-foreground">Setup hint</p>
          <p>
            {!hasModelProviders && "Add a Model Provider block. "}
            {!hasModelConfigs && "Add a Model Config block and pick its alias here."}
          </p>
        </div>
      )}
      <div className="grid gap-2">
        <FieldLabel
          label="Model alias"
          htmlFor={modelAliasId}
          hint="Alias must match a Model Config block."
        />
        <div ref={modelAliasAnchorRef}>
          <Combobox
            items={modelConfigAliases}
            filteredItems={modelConfigAliases}
            filter={null}
            value={config.model_alias || null}
            onValueChange={(value) => onUpdate({ model_alias: value ?? "" })}
            itemToStringValue={(value) => value}
            autoHighlight={true}
          >
            <ComboboxInput
              id={modelAliasId}
              className="nodrag w-full"
              placeholder="Pick model alias or type"
              onBlur={(event) => {
                const inputValue = event.target.value;
                if (inputValue !== config.model_alias) {
                  onUpdate({ model_alias: inputValue });
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
      </div>
      {config.llm_type === "code" && (
        <div className="grid gap-2">
          <FieldLabel
            label="Code language"
            htmlFor={codeLangId}
            hint="Target language for LLM code generation."
          />
          <Select
            value={config.code_lang ?? "python"}
            onValueChange={(value) => onUpdate({ code_lang: value })}
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
        <FieldLabel
          label="Prompt"
          htmlFor={promptId}
          hint="Jinja template. references other columns via {{ variable }}."
        />
        <Textarea
          id={promptId}
          className="corner-squircle nodrag max-h-[600px] overflow-auto"
          value={config.prompt}
          onChange={(event) => onUpdate({ prompt: event.target.value })}
        />
      </div>
      {config.llm_type === "structured" && (
        <div className="grid gap-2">
          <FieldLabel
            label="Output format (JSON schema)"
            htmlFor={outputFormatId}
            hint="Schema used to constrain structured JSON output."
          />
          <Textarea
            id={outputFormatId}
            className="corner-squircle nodrag"
            value={config.output_format ?? ""}
            onChange={(event) =>
              onUpdate({ output_format: event.target.value })
            }
          />
        </div>
      )}
      <div className="grid gap-2">
        <FieldLabel
          label="System prompt (optional)"
          htmlFor={systemPromptId}
          hint="Global behavior instructions prepended before prompt."
        />
        <Textarea
          id={systemPromptId}
          className="corner-squircle nodrag max-h-[600px] overflow-auto"
          value={config.system_prompt}
          onChange={(event) => onUpdate({ system_prompt: event.target.value })}
        />
      </div>
    </div>
  );
}
