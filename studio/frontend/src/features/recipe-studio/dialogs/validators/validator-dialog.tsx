// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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
import { type ReactElement, useMemo, useRef } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { ValidatorConfig } from "../../types";
import {
  isValidatorCodeLang,
  VALIDATOR_OXC_CODE_LANGS,
  VALIDATOR_SQL_CODE_LANGS,
} from "../../utils/validators/code-lang";
import {
  OXC_CODE_SHAPES,
  normalizeOxcCodeShape,
} from "../../utils/validators/oxc-code-shape";
import {
  OXC_VALIDATION_MODES,
  normalizeOxcValidationMode,
} from "../../utils/validators/oxc-mode";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
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
  const oxcModeId = `${config.id}-oxc-mode`;
  const oxcCodeShapeId = `${config.id}-oxc-code-shape`;
  const batchSizeId = `${config.id}-batch-size`;
  const oxcModeAnchorRef = useRef<HTMLDivElement>(null);
  const oxcCodeShapeAnchorRef = useRef<HTMLDivElement>(null);
  const advancedOpen = config.advancedOpen === true;
  const selectedOxcMode = normalizeOxcValidationMode(config.oxc_validation_mode);
  const selectedOxcCodeShape = normalizeOxcCodeShape(config.oxc_code_shape);
  const codeOptions = useMemo(
    () =>
      Object.values(configs)
        .flatMap((item) => {
          if (!(item.kind === "llm" && item.llm_type === "code")) {
            return [];
          }
          if (config.validator_type === "oxc") {
            const lang = item.code_lang?.trim() ?? "";
            if (!VALIDATOR_OXC_CODE_LANGS.includes(lang as typeof config.code_lang)) {
              return [];
            }
          } else {
            const lang = item.code_lang?.trim() ?? "";
            if (
              !(
                lang === "python" ||
                VALIDATOR_SQL_CODE_LANGS.includes(lang as typeof config.code_lang)
              )
            ) {
              return [];
            }
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

  return (
    <div className="space-y-4">
      <NameField
        label="Check name"
        hint="Name used for this check in the canvas and run results."
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-1.5">
        <FieldLabel
          label="Code to check"
          htmlFor={targetColumnId}
          hint="Choose the AI code step this check should review."
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
                {config.validator_type === "oxc"
                  ? "Add an AI code step that generates JavaScript or TypeScript first."
                  : "Add an AI code step first."}
              </p>
        )}
      </div>
      {config.validator_type === "oxc" && (
        <div className="grid gap-3">
          <div className="grid gap-1.5">
            <FieldLabel
              label="Check mode"
              htmlFor={oxcModeId}
              hint="Choose whether to check syntax, lint rules, or both."
            />
            <div ref={oxcModeAnchorRef}>
              <Combobox
                items={OXC_VALIDATION_MODES}
                filteredItems={OXC_VALIDATION_MODES}
                filter={null}
                value={selectedOxcMode}
                onValueChange={(value) =>
                  onUpdate({
                    oxc_validation_mode: normalizeOxcValidationMode(value),
                  })
                }
                itemToStringValue={(value) => value}
                autoHighlight={true}
              >
                <ComboboxInput
                  id={oxcModeId}
                  className="nodrag w-full"
                  placeholder="Select validation mode"
                  readOnly={true}
                />
                <ComboboxContent anchor={oxcModeAnchorRef}>
                  <ComboboxEmpty>No modes available</ComboboxEmpty>
                  <ComboboxList>
                    {(mode: string) => (
                      <ComboboxItem key={mode} value={mode}>
                        {mode}
                      </ComboboxItem>
                    )}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>
            </div>
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Code shape"
              htmlFor={oxcCodeShapeId}
              hint="Choose whether the code should be treated like a full file or a smaller snippet."
            />
            <div ref={oxcCodeShapeAnchorRef}>
              <Combobox
                items={OXC_CODE_SHAPES}
                filteredItems={OXC_CODE_SHAPES}
                filter={null}
                value={selectedOxcCodeShape}
                onValueChange={(value) =>
                  onUpdate({
                    oxc_code_shape: normalizeOxcCodeShape(value),
                  })
                }
                itemToStringValue={(value) => value}
                autoHighlight={true}
              >
                <ComboboxInput
                  id={oxcCodeShapeId}
                  className="nodrag w-full"
                  placeholder="Select code shape"
                  readOnly={true}
                />
                <ComboboxContent anchor={oxcCodeShapeAnchorRef}>
                  <ComboboxEmpty>No code-shape options</ComboboxEmpty>
                  <ComboboxList>
                    {(shape: string) => (
                      <ComboboxItem key={shape} value={shape}>
                        {shape}
                      </ComboboxItem>
                    )}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>
            </div>
          </div>
        </div>
      )}
      <Collapsible
        open={advancedOpen}
        onOpenChange={(open) => onUpdate({ advancedOpen: open })}
      >
        <CollapsibleTrigger asChild={true}>
          <CollapsibleSectionTriggerButton
            label="Advanced check settings"
            open={advancedOpen}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3">
          <div className="grid gap-1.5">
            <FieldLabel
              label="Batch size"
              htmlFor={batchSizeId}
              hint="How many records to check at a time."
            />
            <Input
              id={batchSizeId}
              className="nodrag"
              value={config.batch_size}
              onChange={(event) => onUpdate({ batch_size: event.target.value })}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
