import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type ReactElement, useMemo } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { ValidatorConfig } from "../../types";
import { isValidatorCodeLang } from "../../utils/validators/code-lang";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type ValidatorDialogProps = {
  config: ValidatorConfig;
  onUpdate: (patch: Partial<ValidatorConfig>) => void;
};

const NONE_VALUE = "__none__";

export function ValidatorDialog({
  config,
  onUpdate,
}: ValidatorDialogProps): ReactElement {
  const configs = useRecipeStudioStore((state) => state.configs);
  const targetColumnId = `${config.id}-target-column`;
  const batchSizeId = `${config.id}-batch-size`;
  const codeOptions = useMemo(
    () =>
      Object.values(configs)
        .flatMap((item) => {
          if (!(item.kind === "llm" && item.llm_type === "code")) {
            return [];
          }
          return [
            {
              name: item.name,
              codeLang: item.code_lang?.trim() ?? "",
            },
          ];
        })
        .filter((item) => item.name.trim())
        .sort((a, b) => a.name.localeCompare(b.name)),
    [configs],
  );
  const currentTarget = config.target_columns[0] ?? "";
  const engineLabel = config.code_lang.startsWith("sql:") ? "SQL" : "Python";

  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-2">
        <FieldLabel
          label="Validator engine"
          hint="Built-in validator type for this block."
        />
        <Input value={engineLabel} disabled={true} className="nodrag" />
      </div>
      <div className="grid gap-2">
        <FieldLabel
          label="Target code column"
          htmlFor={targetColumnId}
          hint="Must reference an LLM Code block."
        />
        <Select
          value={currentTarget || NONE_VALUE}
          onValueChange={(value) => {
            if (value === NONE_VALUE) {
              onUpdate({
                // biome-ignore lint/style/useNamingConvention: api schema
                target_columns: [],
              });
              return;
            }
            const targetConfig = codeOptions.find((item) => item.name === value);
            const nextCodeLang = targetConfig?.codeLang?.trim();
            onUpdate({
              // biome-ignore lint/style/useNamingConvention: api schema
              target_columns: [value],
              // biome-ignore lint/style/useNamingConvention: api schema
              code_lang:
                nextCodeLang && isValidatorCodeLang(nextCodeLang)
                  ? nextCodeLang
                  : config.code_lang,
            });
          }}
        >
          <SelectTrigger className="nodrag w-full" id={targetColumnId}>
            <SelectValue placeholder="Select code column" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NONE_VALUE}>None</SelectItem>
            {codeOptions.map((item) => (
              <SelectItem key={item.name} value={item.name}>
                {item.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {codeOptions.length === 0 && (
          <p className="text-xs text-muted-foreground">
            Add an LLM Code block first.
          </p>
        )}
      </div>
      <div className="grid gap-2">
        <FieldLabel
          label="Batch size"
          htmlFor={batchSizeId}
          hint="Records per validation batch."
        />
        <Input
          id={batchSizeId}
          className="nodrag"
          value={config.batch_size}
          onChange={(event) => onUpdate({ batch_size: event.target.value })}
        />
      </div>
    </div>
  );
}
