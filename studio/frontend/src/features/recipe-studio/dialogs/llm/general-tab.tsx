import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, type RefObject, useMemo } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import { isLikelyImageValue } from "../../utils/image-preview";
import type { LlmConfig } from "../../types";
import { findInvalidJinjaReferences } from "../../utils/refs";
import { getAvailableVariables } from "../../utils/variables";
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
  const configs = useRecipeStudioStore((state) => state.configs);
  const modelAliasId = `${config.id}-model-alias`;
  const codeLangId = `${config.id}-code-lang`;
  const promptId = `${config.id}-prompt`;
  const outputFormatId = `${config.id}-output-format`;
  const systemPromptId = `${config.id}-system-prompt`;
  const hasModelConfigs = modelConfigAliases.length > 0;
  const hasModelProviders = modelProviderOptions.length > 0;
  const validReferences = useMemo(
    () => getAvailableVariables(configs, config.id),
    [configs, config.id],
  );
  const invalidPromptRefs = useMemo(
    () => findInvalidJinjaReferences(config.prompt, validReferences),
    [config.prompt, validReferences],
  );
  const invalidSystemRefs = useMemo(
    () => findInvalidJinjaReferences(config.system_prompt, validReferences),
    [config.system_prompt, validReferences],
  );
  const invalidPromptText = invalidPromptRefs
    .slice(0, 3)
    .map((ref) => `{{ ${ref} }}`)
    .join(", ");
  const invalidSystemText = invalidSystemRefs
    .slice(0, 3)
    .map((ref) => `{{ ${ref} }}`)
    .join(", ");
  const seedConfig = useMemo(
    () => Object.values(configs).find((item) => item.kind === "seed"),
    [configs],
  );
  const seedColumns = seedConfig?.seed_columns ?? [];
  const seedPreviewRows = seedConfig?.seed_preview_rows ?? [];
  const imageColumnOptions = useMemo(() => {
    if (seedColumns.length === 0) {
      return [];
    }
    const detected = seedColumns.filter((columnName) => {
      const lower = columnName.toLowerCase();
      if (
        lower.includes("image") ||
        lower.includes("img") ||
        lower.includes("photo") ||
        lower.includes("picture") ||
        lower.includes("base64") ||
        lower.includes("url")
      ) {
        return true;
      }
      return seedPreviewRows.some((row) => isLikelyImageValue(row[columnName]));
    });
    return detected.length > 0 ? detected : seedColumns;
  }, [seedColumns, seedPreviewRows]);
  const imageContext = config.image_context ?? {
    enabled: false,
    // biome-ignore lint/style/useNamingConvention: api schema
    column_name: "",
  };
  const imageContextToggleId = `${config.id}-image-context-enabled`;
  const imageContextColumnId = `${config.id}-image-context-column`;

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
          className="corner-squircle nodrag max-h-[450px] overflow-auto"
          aria-invalid={invalidPromptRefs.length > 0}
          value={config.prompt}
          onChange={(event) => onUpdate({ prompt: event.target.value })}
        />
        {invalidPromptRefs.length > 0 && (
          <p className="text-xs text-destructive">
            Unknown reference: {invalidPromptText}
            {invalidPromptRefs.length > 3
              ? ` +${invalidPromptRefs.length - 3} more`
              : ""}
          </p>
        )}
      </div>
      <div className="space-y-3 rounded-2xl border border-border/60 px-3 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <FieldLabel
              label="Use image context"
              htmlFor={imageContextToggleId}
              hint="Attach one seed image column to this LLM call."
            />
            {imageColumnOptions.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Suggested image columns: {imageColumnOptions.join(", ")}
              </p>
            )}
          </div>
          <Switch
            id={imageContextToggleId}
            checked={imageContext.enabled}
            onCheckedChange={(checked) => {
              onUpdate({
                image_context: {
                  ...imageContext,
                  enabled: checked,
                  // biome-ignore lint/style/useNamingConvention: api schema
                  column_name:
                    checked && !imageContext.column_name
                      ? (imageColumnOptions[0] ?? "")
                      : imageContext.column_name,
                },
              });
            }}
          />
        </div>
        {imageContext.enabled && (
          <div className="grid gap-2">
            <FieldLabel
              label="Image column"
              htmlFor={imageContextColumnId}
              hint="Seed column containing image values."
            />
            <Select
              value={imageContext.column_name || ""}
              onValueChange={(value) =>
                onUpdate({
                  image_context: {
                    ...imageContext,
                    // biome-ignore lint/style/useNamingConvention: api schema
                    column_name: value,
                  },
                })
              }
            >
              <SelectTrigger className="nodrag w-full" id={imageContextColumnId}>
                <SelectValue placeholder="Select image column" />
              </SelectTrigger>
              <SelectContent>
                {imageColumnOptions.map((columnName) => (
                  <SelectItem key={columnName} value={columnName}>
                    {columnName}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
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
          className="corner-squircle nodrag max-h-[450px] overflow-auto"
          aria-invalid={invalidSystemRefs.length > 0}
          value={config.system_prompt}
          onChange={(event) => onUpdate({ system_prompt: event.target.value })}
        />
        {invalidSystemRefs.length > 0 && (
          <p className="text-xs text-destructive">
            Unknown reference: {invalidSystemText}
            {invalidSystemRefs.length > 3
              ? ` +${invalidSystemRefs.length - 3} more`
              : ""}
          </p>
        )}
      </div>
    </div>
  );
}
