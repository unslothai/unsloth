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
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import { getAvailableRefItems } from "../../utils/variables";
import { JinjaRefTextarea } from "../../components/jinja/jinja-ref-autocomplete";
import { AvailableVariables } from "../shared/available-variables";
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
  modelAliasAnchorRef: RefObject<HTMLDivElement | null>;
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

export function LlmGeneralTab({
  config,
  modelConfigAliases,
  modelAliasAnchorRef,
  onUpdate,
}: LlmGeneralTabProps): ReactElement {
  const configs = useRecipeStudioStore((state) => state.configs);
  const items = getAvailableRefItems(configs, config.id);
  const modelAliasId = `${config.id}-model-alias`;
  const codeLangId = `${config.id}-code-lang`;
  const promptId = `${config.id}-prompt`;
  const outputFormatId = `${config.id}-output-format`;
  const systemPromptId = `${config.id}-system-prompt`;

  return (
    <div className="space-y-4">
      <AvailableVariables configId={config.id} />
      <NameField value={config.name} onChange={(value) => onUpdate({ name: value })} />
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
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor={codeLangId}
          >
            Code language
          </label>
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
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={promptId}
        >
          Prompt
        </label>
        <JinjaRefTextarea
          id={promptId}
          className="corner-squircle nodrag"
          value={config.prompt}
          items={items}
          onValueChange={(value) => onUpdate({ prompt: value })}
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
            className="corner-squircle nodrag"
            value={config.output_format ?? ""}
            onChange={(event) =>
              onUpdate({ output_format: event.target.value })
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
        <JinjaRefTextarea
          id={systemPromptId}
          className="corner-squircle nodrag"
          value={config.system_prompt}
          items={items}
          onValueChange={(value) => onUpdate({ system_prompt: value })}
        />
      </div>
    </div>
  );
}
