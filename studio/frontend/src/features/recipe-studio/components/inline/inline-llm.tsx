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
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { LlmConfig } from "../../types";
import { InlineField } from "./inline-field";

type InlineLlmProps = {
  config: LlmConfig;
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

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
] as const;

export function InlineLlm({ config, onUpdate }: InlineLlmProps): ReactElement {
  const isCode = config.llm_type === "code";
  const configs = useRecipeStudioStore((state) => state.configs);
  const modelConfigAliases = useMemo(
    () =>
      Object.values(configs)
        .filter((c) => c.kind === "model_config")
        .map((c) => c.name),
    [configs],
  );
  const [aliasInput, setAliasInput] = useState(config.model_alias);
  const anchorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setAliasInput(config.model_alias);
  }, [config.model_alias]);

  return (
    <div className="space-y-3">
      <InlineField label="Model alias">
        <div ref={anchorRef}>
          <Combobox
            items={modelConfigAliases}
            filteredItems={modelConfigAliases}
            filter={null}
            value={config.model_alias || null}
            onValueChange={(value) =>
              onUpdate({
                // biome-ignore lint/style/useNamingConvention: api schema
                model_alias: value ?? "",
              })
            }
            onInputValueChange={setAliasInput}
            itemToStringValue={(value) => value}
            autoHighlight={true}
          >
            <ComboboxInput
              className="nodrag h-8 w-full text-xs"
              placeholder="Model alias"
              onBlur={() => {
                if (aliasInput !== config.model_alias) {
                  onUpdate({
                    // biome-ignore lint/style/useNamingConvention: api schema
                    model_alias: aliasInput,
                  });
                }
              }}
            />
            <ComboboxContent anchor={anchorRef}>
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
      </InlineField>
      {isCode && (
        <InlineField label="Code language">
          <Select
            value={config.code_lang?.trim() || "python"}
            onValueChange={(value) =>
              onUpdate({
                // biome-ignore lint/style/useNamingConvention: api schema
                code_lang: value,
              })
            }
          >
            <SelectTrigger className="nodrag h-8 w-full text-xs">
              <SelectValue placeholder="Language" />
            </SelectTrigger>
            <SelectContent>
              {CODE_LANG_OPTIONS.map((lang) => (
                <SelectItem key={lang} value={lang}>
                  {lang}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </InlineField>
      )}
      <p className="text-[11px] text-muted-foreground">
        Prompt/system edited on aux nodes.
      </p>
    </div>
  );
}
