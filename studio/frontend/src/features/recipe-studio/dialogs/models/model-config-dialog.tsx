// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, useEffect, useRef, useState } from "react";
import { useI18n } from "@/features/i18n";
import type { ModelConfig } from "../../types";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type ModelConfigDialogProps = {
  config: ModelConfig;
  providerOptions: string[];
  localProviderNames: Set<string>;
  onUpdate: (patch: Partial<ModelConfig>) => void;
};

export function ModelConfigDialog({
  config,
  providerOptions,
  localProviderNames,
  onUpdate,
}: ModelConfigDialogProps): ReactElement {
  const { t } = useI18n();
  const isLinkedToLocal = localProviderNames.has(config.provider);
  const [optionalOpen, setOptionalOpen] = useState(false);
  const modelId = `${config.id}-model`;
  const providerId = `${config.id}-provider`;
  const tempId = `${config.id}-temperature`;
  const topPId = `${config.id}-top-p`;
  const maxTokensId = `${config.id}-max-tokens`;
  const timeoutId = `${config.id}-timeout`;
  const extraBodyId = `${config.id}-inference-extra-body`;
  const providerAnchorRef = useRef<HTMLDivElement>(null);
  const providerInputRef = useRef(config.provider);
  // Sync providerInputRef with the current provider value. Updating a ref in
  // an effect (vs reading/writing it during render) satisfies the
  // react-hooks/refs rule and keeps the combobox blur path stable across
  // re-renders.
  useEffect(() => {
    providerInputRef.current = config.provider;
  }, [config.provider]);
  const updateField = <K extends keyof ModelConfig>(
    key: K,
    value: ModelConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<ModelConfig>);
  };

  // Apply provider selection while keeping the local-provider model autofill
  // consistent across both dropdown selection and free-typed + blur input.
  const applyProviderChange = (selectedProvider: string) => {
    const isLocal = localProviderNames.has(selectedProvider);
    if (isLocal && !config.model.trim()) {
      onUpdate({ provider: selectedProvider, model: "local" });
      return;
    }
    if (!isLocal && config.model === "local") {
      onUpdate({ provider: selectedProvider, model: "" });
      return;
    }
    updateField("provider", selectedProvider);
  };

  return (
    <div className="space-y-4">
      <NameField
        label={t("recipe.modelConfig.presetName")}
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="rounded-2xl border border-border/60 bg-muted/10 px-4 py-3">
        <p className="text-sm font-semibold text-foreground">
          {t("recipe.modelConfig.setupTitle")}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          {t("recipe.modelConfig.setupDescription")}
        </p>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label={t("recipe.modelConfig.providerConnection")}
          htmlFor={providerId}
          hint={t("recipe.modelConfig.providerConnectionHint")}
        />
        <div ref={providerAnchorRef}>
          <Combobox
            items={providerOptions}
            filteredItems={providerOptions}
            filter={null}
            value={config.provider || null}
            onValueChange={(value) => applyProviderChange(value ?? "")}
            onInputValueChange={(value) => {
              providerInputRef.current = value;
            }}
            itemToStringValue={(value) => value}
            autoHighlight={true}
          >
            <ComboboxInput
              id={providerId}
              className="nodrag w-full"
              placeholder={t("recipe.modelConfig.providerPlaceholder")}
              onBlur={() => {
                const next = providerInputRef.current;
                if (next !== config.provider) {
                  applyProviderChange(next);
                }
              }}
            />
            <ComboboxContent anchor={providerAnchorRef}>
              <ComboboxEmpty>{t("recipe.modelConfig.noProviders")}</ComboboxEmpty>
              <ComboboxList>
                {(provider: string) => (
                  <ComboboxItem key={provider} value={provider}>
                    {provider}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </div>
        <p className="text-xs text-muted-foreground">
          {providerOptions.length === 0
            ? t("recipe.modelConfig.providerEmptyHint")
            : t("recipe.modelConfig.providerLinkedHint")}
        </p>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label={t("recipe.modelConfig.modelId")}
          htmlFor={modelId}
          hint={
            isLinkedToLocal
              ? t("recipe.modelConfig.modelIdHintLocal")
              : t("recipe.modelConfig.modelIdHintExternal")
          }
        />
        <Input
          id={modelId}
          className="nodrag"
          placeholder={isLinkedToLocal ? "local" : t("recipe.modelConfig.modelIdPlaceholder")}
          value={config.model}
          onChange={(event) => updateField("model", event.target.value)}
        />
      </div>
      <div className="grid gap-3">
        <div className="space-y-1">
          <p className="text-sm font-semibold text-foreground">
            {t("recipe.modelConfig.defaultsTitle")}
          </p>
          <p className="text-xs text-muted-foreground">
            {t("recipe.modelConfig.defaultsDescription")}
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="grid gap-1.5">
            <FieldLabel
              label={t("recipe.modelConfig.temperature")}
              htmlFor={tempId}
              hint={t("recipe.modelConfig.temperatureHint")}
            />
            <Input
              id={tempId}
              className="nodrag"
              value={config.inference_temperature ?? ""}
              onChange={(event) =>
                updateField("inference_temperature", event.target.value)
              }
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label={t("recipe.modelConfig.topP")}
              htmlFor={topPId}
              hint={t("recipe.modelConfig.topPHint")}
            />
            <Input
              id={topPId}
              className="nodrag"
              value={config.inference_top_p ?? ""}
              onChange={(event) =>
                updateField("inference_top_p", event.target.value)
              }
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label={t("recipe.modelConfig.maxTokens")}
              htmlFor={maxTokensId}
              hint={t("recipe.modelConfig.maxTokensHint")}
            />
            <Input
              id={maxTokensId}
              className="nodrag"
              value={config.inference_max_tokens ?? ""}
              onChange={(event) =>
                updateField("inference_max_tokens", event.target.value)
              }
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label={t("recipe.modelConfig.timeout")}
              htmlFor={timeoutId}
              hint={t("recipe.modelConfig.timeoutHint")}
            />
            <Input
              id={timeoutId}
              className="nodrag"
              value={config.inference_timeout ?? ""}
              onChange={(event) =>
                updateField("inference_timeout", event.target.value)
              }
            />
          </div>
        </div>
      </div>
      <Collapsible open={optionalOpen} onOpenChange={setOptionalOpen}>
        <CollapsibleTrigger asChild={true}>
          <CollapsibleSectionTriggerButton
            label={t("recipe.modelConfig.advanced")}
            open={optionalOpen}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-4">
          <div className="grid gap-1.5">
            <FieldLabel
              label={t("recipe.modelConfig.advancedFields")}
              htmlFor={extraBodyId}
              hint={t("recipe.modelConfig.advancedFieldsHint")}
            />
            <Textarea
              id={extraBodyId}
              className="corner-squircle nodrag"
              placeholder='{"top_k": 20, "min_p": 0.0}'
              value={config.inference_extra_body ?? ""}
              onChange={(event) =>
                updateField("inference_extra_body", event.target.value)
              }
            />
          </div>
          <label className="flex items-center gap-2 text-xs font-semibold uppercase text-muted-foreground">
            <Checkbox
              checked={config.skip_health_check ?? false}
              onCheckedChange={(value) =>
                updateField("skip_health_check", Boolean(value))
              }
            />
            {t("recipe.modelConfig.skipConnectionCheck")}
          </label>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
