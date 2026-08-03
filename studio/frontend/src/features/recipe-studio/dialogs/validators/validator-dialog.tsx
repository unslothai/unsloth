// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import {
  Delete02Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, useMemo, useRef } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { ToolScaffoldFile, ValidatorConfig } from "../../types";
import { DEFAULT_CUSTOM_VALIDATOR_SOURCE } from "../../utils/config-factories";
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
import { normalizeToolScaffold } from "../../utils/validators/validation-markers";
import {
  addToolScaffoldRow as addScaffoldRow,
  removeToolScaffoldRow as removeScaffoldRow,
  updateToolScaffoldRow as updateScaffoldRow,
} from "../../utils/validators/tool-scaffold";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type ValidatorDialogProps = {
  config: ValidatorConfig;
  onUpdate: (patch: Partial<ValidatorConfig>) => void;
};

const NONE_VALUE = "__none__";
const LEADING_DOTS_RE = /^\.+/;

const TOOL_EXAMPLES: {
  label: string;
  command: string;
  ext: string;
  scaffold: ToolScaffoldFile[];
}[] = [
  {
    label: "go vet + build (Go)",
    command: "go vet ./... && go build ./...",
    ext: "go",
    scaffold: [
      { path: "go.mod", content: "module example.com/check\n\ngo 1.21\n" },
      { path: "main.go", content: "{source}" },
    ],
  },
  {
    label: "cargo check (Rust)",
    command: "cargo check",
    ext: "rs",
    scaffold: [
      {
        path: "Cargo.toml",
        content: '[package]\nname = "check"\nversion = "0.1.0"\nedition = "2021"\n',
      },
      { path: "src/main.rs", content: "{source}" },
    ],
  },
  {
    label: "SQL lint (Postgres)",
    command: "sqlfluff lint --dialect postgres {file}",
    ext: "sql",
    scaffold: [],
  },
];

const GO_CUSTOM_SAMPLE = String.raw`# Runs go vet and go build on each generated code cell.
def validate(df):
    def run_go(code):
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            (work / "go.mod").write_text("module example.com/check\n\ngo 1.21\n")
            (work / "main.go").write_text(code)
            results = []
            for args in (["go", "vet", "./..."], ["go", "build", "./..."]):
                results.append(
                    subprocess.run(args, cwd=work, capture_output=True, text=True, timeout=60)
                )
            failed = [result for result in results if result.returncode != 0]
            if not failed:
                return {"is_valid": True, "error_message": ""}
            output = "\n".join(
                result.stdout + result.stderr for result in failed
            ).strip()
            return {"is_valid": False, "error_message": output[:300]}

    rows = [run_go(str(value)) for value in df.iloc[:, 0]]
    return pd.DataFrame(rows)
`;

const CARGO_CUSTOM_SAMPLE = String.raw`# Runs cargo check on each generated code cell.
def validate(df):
    def run_cargo(code):
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            (work / "Cargo.toml").write_text(
                '[package]\nname = "check"\nversion = "0.1.0"\nedition = "2021"\n'
            )
            src = work / "src"
            src.mkdir()
            (src / "main.rs").write_text(code)
            result = subprocess.run(
                ["cargo", "check"],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return {"is_valid": True, "error_message": ""}
            output = (result.stdout + result.stderr).strip()
            return {"is_valid": False, "error_message": output[:300]}

    rows = [run_cargo(str(value)) for value in df.iloc[:, 0]]
    return pd.DataFrame(rows)
`;

export function ValidatorDialog({
  config,
  onUpdate,
}: ValidatorDialogProps): ReactElement {
  const configs = useRecipeStudioStore((state) => state.configs);
  const targetColumnId = `${config.id}-target-column`;
  const oxcModeId = `${config.id}-oxc-mode`;
  const oxcCodeShapeId = `${config.id}-oxc-code-shape`;
  const toolCommandId = `${config.id}-tool-command`;
  const toolExtId = `${config.id}-tool-ext`;
  const toolAckId = `${config.id}-tool-ack`;
  const customSourceId = `${config.id}-custom-source`;
  const customAckId = `${config.id}-custom-ack`;
  const batchSizeId = `${config.id}-batch-size`;
  const oxcModeAnchorRef = useRef<HTMLDivElement>(null);
  const oxcCodeShapeAnchorRef = useRef<HTMLDivElement>(null);
  const advancedOpen = config.advancedOpen === true;
  const selectedOxcMode = normalizeOxcValidationMode(config.oxc_validation_mode);
  const selectedOxcCodeShape = normalizeOxcCodeShape(config.oxc_code_shape);
  const acceptsAnyCodeLang =
    config.validator_type === "tool" || config.validator_type === "custom";
  const codeOptions = useMemo(
    () =>
      Object.values(configs)
        .flatMap((item) => {
          if (!(item.kind === "llm" && item.llm_type === "code")) {
            return [];
          }
          if (acceptsAnyCodeLang) {
            return [
              {
                name: item.name,
                codeLang: item.code_lang?.trim() ?? "",
              },
            ];
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
    [configs, config, acceptsAnyCodeLang],
  );
  const currentTarget = config.target_columns[0] ?? "";
  const toolScaffoldRows = Array.isArray(config.tool_scaffold)
    ? config.tool_scaffold
    : [];
  const normalizedToolScaffold = useMemo(
    () => normalizeToolScaffold(config.tool_scaffold),
    [config.tool_scaffold],
  );
  const toolCommandTrimmed = (config.tool_command ?? "").trim();
  const toolExtTrimmed = (config.tool_ext ?? "").trim().replace(LEADING_DOTS_RE, "");
  const toolReferencesFile =
    toolCommandTrimmed.includes("{file}") ||
    (toolExtTrimmed.length > 0 &&
      toolCommandTrimmed.includes(`main.${toolExtTrimmed}`));
  const showToolScaffoldHint =
    normalizedToolScaffold.length === 0 &&
    toolCommandTrimmed.length > 0 &&
    !toolReferencesFile;

  function updateToolScaffoldRow(index: number, next: ToolScaffoldFile): void {
    onUpdate({ tool_scaffold: updateScaffoldRow(toolScaffoldRows, index, next) });
  }

  function removeToolScaffoldRow(index: number): void {
    onUpdate({ tool_scaffold: removeScaffoldRow(toolScaffoldRows, index) });
  }

  function addToolScaffoldRow(): void {
    onUpdate({ tool_scaffold: addScaffoldRow(toolScaffoldRows) });
  }

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
      {config.validator_type === "tool" && (
        <div className="grid gap-4">
          <div className="grid gap-1.5">
            <FieldLabel
              label="Examples"
              hint="Try one of these examples, then edit the command, extension and files to suit."
            />
            <div className="flex flex-wrap gap-1.5">
              {TOOL_EXAMPLES.map((example) => (
                <Button
                  key={example.label}
                  type="button"
                  variant="outline"
                  size="sm"
                  className="nodrag"
                  onClick={() =>
                    onUpdate({
                      tool_command: example.command,
                      tool_ext: example.ext,
                      tool_scaffold: example.scaffold.map((file) => ({ ...file })),
                    })
                  }
                >
                  {example.label}
                </Button>
              ))}
            </div>
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Tool command"
              htmlFor={toolCommandId}
              hint="Use {file} for the generated source file and {dir} for its temp folder. Commands run from that temp folder, so relative paths like ./... already cover the generated file. Quote the placeholders if your temp path contains spaces."
            />
            <Textarea
              id={toolCommandId}
              className="nodrag font-mono"
              fieldSizing="content"
              value={config.tool_command ?? ""}
              onChange={(event) =>
                onUpdate({ tool_command: event.target.value })
              }
              placeholder="go vet ./..."
            />
            {showToolScaffoldHint && (
              <p className="text-xs text-muted-foreground">
                Your command doesn&apos;t reference the generated file (
                {"{file}"}). Checks that require a project scaffolding may fail
                every row at runtime.
              </p>
            )}
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="File extension"
              htmlFor={toolExtId}
              hint="Generated code is written to a temp file with this extension unless a Files to write entry contains {source}."
            />
            <Input
              id={toolExtId}
              className="nodrag font-mono"
              value={config.tool_ext ?? ""}
              onChange={(event) => onUpdate({ tool_ext: event.target.value })}
              placeholder="go"
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Files to write"
              hint="Optional files written into the check's temp folder before the command runs. Paths are relative to that folder — write the full path yourself, so src/main.rs lands in a src/ subfolder (created automatically). A file whose content contains {source} receives the generated code, and {file} then points at it."
            />
            <div className="grid gap-2">
              {toolScaffoldRows.map((file, index) => (
                <div
                  key={index}
                  className="grid gap-1.5 rounded-xl border border-border/50 p-2"
                >
                  <div className="flex items-center gap-2">
                    <Input
                      className="nodrag font-mono"
                      value={file.path}
                      onChange={(event) =>
                        updateToolScaffoldRow(index, {
                          ...file,
                          path: event.target.value,
                        })
                      }
                      placeholder="src/main.rs"
                    />
                    <Button
                      type="button"
                      size="icon-sm"
                      variant="ghost"
                      className="nodrag shrink-0"
                      onClick={() => removeToolScaffoldRow(index)}
                    >
                      <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                    </Button>
                  </div>
                  <Textarea
                    className="nodrag font-mono"
                    fieldSizing="content"
                    value={file.content}
                    onChange={(event) =>
                      updateToolScaffoldRow(index, {
                        ...file,
                        content: event.target.value,
                      })
                    }
                    placeholder="{source}"
                  />
                </div>
              ))}
              <Button
                type="button"
                size="xs"
                variant="outline"
                className="nodrag justify-self-start"
                onClick={addToolScaffoldRow}
              >
                <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
                Add file
              </Button>
            </div>
          </div>
          <div className="grid gap-2">
            <label
              htmlFor={toolAckId}
              className="flex cursor-pointer items-start gap-1.5 text-xs"
            >
              <Checkbox
                id={toolAckId}
                checked={config.tool_acknowledged === true}
                onCheckedChange={(value) =>
                  onUpdate({ tool_acknowledged: value === true })
                }
              />
              <span>
                I understand this check runs an arbitrary command on my machine.
              </span>
            </label>
            <p className="text-xs text-destructive">
              This runs locally in the job worker. Only add checks you trust.
            </p>
          </div>
        </div>
      )}
      {config.validator_type === "custom" && (
        <div className="grid gap-4">
          <div className="grid gap-1.5">
            <FieldLabel
              label="Python function"
              htmlFor={customSourceId}
              hint="Define validate(df) -> df returning a DataFrame with a boolean is_valid column. pd, subprocess, tempfile and Path are pre-imported; only add imports for modules specific to your code."
            />
            <Textarea
              id={customSourceId}
              className="nodrag min-h-64 font-mono"
              fieldSizing="content"
              value={config.custom_source ?? ""}
              onChange={(event) =>
                onUpdate({ custom_source: event.target.value })
              }
            />
            <div className="flex flex-wrap gap-1.5">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="nodrag"
                onClick={() => onUpdate({ custom_source: GO_CUSTOM_SAMPLE })}
              >
                Insert Go vet + build sample
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="nodrag"
                onClick={() =>
                  onUpdate({ custom_source: CARGO_CUSTOM_SAMPLE })
                }
              >
                Insert cargo check sample
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="nodrag"
                onClick={() =>
                  onUpdate({ custom_source: DEFAULT_CUSTOM_VALIDATOR_SOURCE })
                }
              >
                Reset
              </Button>
            </div>
          </div>
          <div className="grid gap-2">
            <label
              htmlFor={customAckId}
              className="flex cursor-pointer items-start gap-1.5 text-xs"
            >
              <Checkbox
                id={customAckId}
                checked={config.custom_acknowledged === true}
                onCheckedChange={(value) =>
                  onUpdate({ custom_acknowledged: value === true })
                }
              />
              <span>
                I understand this check runs arbitrary Python on my machine.
              </span>
            </label>
            <p className="text-xs text-destructive">
              This runs locally in the job worker. Only add code you trust.
            </p>
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
              hint={
                config.validator_type === "tool"
                  ? "How many records to check at a time. Tool checks run up to this many in parallel, capped at the detected CPU count; the default of 1 runs each check one at a time."
                  : "How many records to check at a time."
              }
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
