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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, type RefObject, useMemo, useRef } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { LlmConfig } from "../../types";
import { isLikelyImageValue } from "../../utils/image-preview";
import { findInvalidJinjaReferences } from "../../utils/refs";
import { getAvailableVariables } from "../../utils/variables";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
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

const TRACE_MODE_OPTIONS = ["none", "last_message", "all_messages"] as const;

function normalizeTraceMode(value: string): LlmConfig["with_trace"] {
  if (value === "last_message" || value === "all_messages") {
    return value;
  }
  return "none";
}

type LlmGeneralTabProps = {
  config: LlmConfig;
  modelConfigAliases: string[];
  modelProviderOptions: string[];
  toolProfileAliases: string[];
  modelAliasAnchorRef: RefObject<HTMLDivElement | null>;
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

export function LlmGeneralTab({
  config,
  modelConfigAliases,
  modelProviderOptions,
  toolProfileAliases,
  modelAliasAnchorRef,
  onUpdate,
}: LlmGeneralTabProps): ReactElement {
  const configs = useRecipeStudioStore((state) => state.configs);
  const modelAliasId = `${config.id}-model-alias`;
  const toolAliasId = `${config.id}-tool-alias`;
  const codeLangId = `${config.id}-code-lang`;
  const promptId = `${config.id}-prompt`;
  const outputFormatId = `${config.id}-output-format`;
  const systemPromptId = `${config.id}-system-prompt`;
  const hasModelConfigs = modelConfigAliases.length > 0;
  const hasModelProviders = modelProviderOptions.length > 0;
  const hasToolProfiles = toolProfileAliases.length > 0;
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
  const hasHfSeed = Boolean(
    seedConfig && (seedConfig.seed_source_type ?? "hf") === "hf",
  );
  const seedColumns = useMemo(
    () => seedConfig?.seed_columns ?? [],
    [seedConfig],
  );
  const seedPreviewRows = useMemo(
    () => seedConfig?.seed_preview_rows ?? [],
    [seedConfig],
  );
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
  const imageContextColumnOptions = useMemo(() => {
    const preferred =
      imageColumnOptions.length > 0 ? imageColumnOptions : seedColumns;
    const deduped = Array.from(
      new Set(preferred.map((value) => value.trim()).filter(Boolean)),
    );
    const selected = imageContext.column_name.trim();
    if (selected && !deduped.includes(selected)) {
      deduped.unshift(selected);
    }
    return deduped;
  }, [imageColumnOptions, imageContext.column_name, seedColumns]);
  const traceModeId = `${config.id}-trace-mode`;
  const reasoningToggleId = `${config.id}-reasoning-content`;
  const advancedOpen = config.advancedOpen === true;
  const toolAliasAnchorRef = useRef<HTMLDivElement>(null);
  const needsSetupHelp = !hasModelConfigs || !hasModelProviders;
  const needsModelChoice = !config.model_alias?.trim();

  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      {needsSetupHelp ? (
        <div className="rounded-2xl border border-border/60 bg-muted/10 px-4 py-3 text-xs text-muted-foreground">
          <p className="text-sm font-semibold text-foreground">
            先完成模型配置，再回到这里
          </p>
          <div className="mt-2 space-y-1.5">
            {!hasModelProviders && (
              <p className="flex items-start gap-2">
                <HugeiconsIcon
                  icon={ArrowRight01Icon}
                  className="mt-0.5 size-3.5 shrink-0 text-primary"
                />
                <span>在 AI generation → Setup 中先添加提供方连接步骤。</span>
              </p>
            )}
            {!hasModelConfigs && (
              <p className="flex items-start gap-2">
                <HugeiconsIcon
                  icon={ArrowRight01Icon}
                  className="mt-0.5 size-3.5 shrink-0 text-primary"
                />
                <span>先添加模型预设步骤并连接，再在下方选择。</span>
              </p>
            )}
          </div>
        </div>
      ) : needsModelChoice ? (
        <div className="rounded-2xl border border-border/60 bg-muted/10 px-4 py-3 text-xs text-muted-foreground">
          <p className="text-sm font-semibold text-foreground">
            请先选择模型预设
          </p>
          <p className="mt-1">
            选好后可编写提示词；若本步骤需要工具，再配置工具权限。
          </p>
        </div>
      ) : null}
      <div className="grid gap-1.5">
        <FieldLabel
          label="模型预设"
          htmlFor={modelAliasId}
          hint="选择当前步骤复用的模型配置。"
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
                placeholder="选择模型预设"
              onBlur={(event) => {
                const inputValue = event.target.value;
                if (inputValue !== config.model_alias) {
                  onUpdate({ model_alias: inputValue });
                }
              }}
            />
            <ComboboxContent anchor={modelAliasAnchorRef}>
              <ComboboxEmpty>未找到模型配置</ComboboxEmpty>
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
      {!hasToolProfiles && (
        <p className="text-xs text-muted-foreground">
          如需工具，请在 AI generation → Setup 中添加 Tool access 步骤。
        </p>
      )}
      {(hasToolProfiles || Boolean(config.tool_alias?.trim())) && (
        <div className="grid gap-1.5">
          <FieldLabel
            label="工具权限（可选）"
            htmlFor={toolAliasId}
            hint="选择该步骤可使用的工具权限；不需要工具时可留空。"
          />
          <div ref={toolAliasAnchorRef}>
            <Combobox
              items={toolProfileAliases}
              filteredItems={toolProfileAliases}
              filter={null}
              value={config.tool_alias || null}
              onValueChange={(value) => onUpdate({ tool_alias: value ?? "" })}
              itemToStringValue={(value) => value}
              autoHighlight={true}
            >
              <ComboboxInput
                id={toolAliasId}
                className="nodrag w-full"
                placeholder="选择工具权限"
                onBlur={(event) => {
                  const inputValue = event.target.value;
                  if (inputValue !== (config.tool_alias ?? "")) {
                    onUpdate({ tool_alias: inputValue });
                  }
                }}
              />
              <ComboboxContent anchor={toolAliasAnchorRef}>
                <ComboboxEmpty>未找到工具权限</ComboboxEmpty>
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
      )}
      {config.llm_type === "code" && (
        <div className="grid gap-1.5">
          <FieldLabel
            label="代码语言"
            htmlFor={codeLangId}
            hint="选择该 AI 步骤要生成的语言。"
          />
          <Select
            value={config.code_lang ?? "python"}
            onValueChange={(value) => onUpdate({ code_lang: value })}
          >
            <SelectTrigger className="nodrag w-full" id={codeLangId}>
              <SelectValue placeholder="选择语言" />
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
      <div className="grid gap-1.5">
        <FieldLabel
          label="提示词"
          htmlFor={promptId}
          hint="编写此步骤提示词，可用 {{ field_name }} 引用其他字段。"
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
            未知字段：{invalidPromptText}
            {invalidPromptRefs.length > 3
              ? ` 另有 ${invalidPromptRefs.length - 3} 个`
              : ""}
          </p>
        )}
      </div>
      <AvailableVariables configId={config.id} />
      {hasHfSeed && (
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-3">
            <FieldLabel
              label="使用图像上下文"
              htmlFor={imageContextToggleId}
              hint="为该 AI 步骤附加来源数据中的图像字段。"
            />
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
                        ? (imageContextColumnOptions[0] ?? "")
                        : imageContext.column_name,
                  },
                });
              }}
            />
          </div>
          {imageContext.enabled && (
            <div className="grid gap-1.5">
              <FieldLabel
                label="图像字段"
                htmlFor={imageContextColumnId}
                hint="选择包含图像内容的来源字段。"
              />
              <Select
                value={imageContext.column_name || undefined}
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
                <SelectTrigger
                  className="nodrag w-full"
                  id={imageContextColumnId}
                >
                  <SelectValue placeholder="选择图像列" />
                </SelectTrigger>
                <SelectContent>
                  {imageContextColumnOptions.map((columnName) => (
                    <SelectItem key={columnName} value={columnName}>
                      {columnName}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      )}
      {config.llm_type === "structured" && (
        <div className="grid gap-1.5">
          <FieldLabel
            label="响应格式"
            htmlFor={outputFormatId}
            hint="描述期望返回的 JSON 结构。"
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
      <Collapsible
        open={advancedOpen}
        onOpenChange={(open) => onUpdate({ advancedOpen: open })}
      >
        <CollapsibleTrigger asChild={true}>
          <CollapsibleSectionTriggerButton
            label="追踪与附加控制"
            open={advancedOpen}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-4">
          <div className="grid gap-1.5">
            <FieldLabel
              label="额外指令（可选）"
              htmlFor={systemPromptId}
              hint="在提示词前补充全局引导信息。"
            />
            <Textarea
              id={systemPromptId}
              className="corner-squircle nodrag max-h-[450px] overflow-auto"
              aria-invalid={invalidSystemRefs.length > 0}
              value={config.system_prompt}
              onChange={(event) =>
                onUpdate({ system_prompt: event.target.value })
              }
            />
            {invalidSystemRefs.length > 0 && (
              <p className="text-xs text-destructive">
                未知字段：{invalidSystemText}
                {invalidSystemRefs.length > 3
                  ? ` 另有 ${invalidSystemRefs.length - 3} 个`
                  : ""}
              </p>
            )}
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="保存追踪详情"
              htmlFor={traceModeId}
              hint="会新增可后续查看的追踪字段。"
            />
            <Select
              value={config.with_trace ?? "none"}
              onValueChange={(value) =>
                onUpdate({
                  // biome-ignore lint/style/useNamingConvention: api schema
                  with_trace: normalizeTraceMode(value),
                })
              }
            >
              <SelectTrigger className="nodrag w-full" id={traceModeId}>
                <SelectValue placeholder="选择追踪模式" />
              </SelectTrigger>
              <SelectContent>
                {TRACE_MODE_OPTIONS.map((traceMode) => (
                  <SelectItem key={traceMode} value={traceMode}>
                    {traceMode}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center justify-between gap-3">
            <FieldLabel
              label="保存推理文本"
              htmlFor={reasoningToggleId}
              hint="当模型返回推理内容时，保存到字段中。"
            />
            <Switch
              id={reasoningToggleId}
              checked={config.extract_reasoning_content === true}
              onCheckedChange={(checked) =>
                onUpdate({
                  // biome-ignore lint/style/useNamingConvention: api schema
                  extract_reasoning_content: checked,
                })
              }
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
