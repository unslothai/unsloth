// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, useEffect, useRef, useState } from "react";
import type { ModelConfig, ModelGeneration } from "../../types";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";
import { LocalRecipeModelSelector } from "./local-recipe-model-selector";

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
  const isLinkedToLocal = localProviderNames.has(config.provider);
  const isImage = config.generation === "image";
  const [optionalOpen, setOptionalOpen] = useState(false);
  const modelId = `${config.id}-model`;
  const providerId = `${config.id}-provider`;
  const tempId = `${config.id}-temperature`;
  const topPId = `${config.id}-top-p`;
  const maxTokensId = `${config.id}-max-tokens`;
  const timeoutId = `${config.id}-timeout`;
  const sizeId = `${config.id}-image-size`;
  const stepsId = `${config.id}-image-steps`;
  const guidanceId = `${config.id}-image-guidance`;
  const extraBodyId = `${config.id}-inference-extra-body`;
  const skipHealthCheckId = `${config.id}-skip-health-check`;
  const providerAnchorRef = useRef<HTMLDivElement>(null);
  const providerInputRef = useRef(config.provider);
  // Sync the ref in an effect, not during render, to satisfy react-hooks/refs.
  useEffect(() => {
    providerInputRef.current = config.provider;
  }, [config.provider]);
  const updateField = <K extends keyof ModelConfig>(
    key: K,
    value: ModelConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<ModelConfig>);
  };

  const applyProviderChange = (selectedProvider: string) => {
    const nextIsLocal = localProviderNames.has(selectedProvider);
    if (isLinkedToLocal !== nextIsLocal) {
      onUpdate({
        provider: selectedProvider,
        model: "",
        // biome-ignore lint/style/useNamingConvention: api schema
        gguf_variant: undefined,
      });
      return;
    }
    if (!nextIsLocal) {
      onUpdate({
        provider: selectedProvider,
        // biome-ignore lint/style/useNamingConvention: api schema
        gguf_variant: undefined,
      });
      return;
    }
    updateField("provider", selectedProvider);
  };

  return (
    <div className="space-y-4">
      <NameField
        label="Model preset name"
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="rounded-lg border border-border/60 bg-muted/10 px-4 py-3">
        <p className="text-sm font-semibold text-foreground">
          Set up one reusable model choice for your AI steps
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          Choose the provider connection, enter the exact model ID, then save
          any generation defaults you want to reuse.
        </p>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Generates"
          hint="Text presets serve AI text steps; image presets serve AI image steps."
        />
        <Tabs
          value={isImage ? "image" : "text"}
          onValueChange={(value) =>
            updateField("generation", value as ModelGeneration)
          }
        >
          <TabsList className="w-full">
            <TabsTrigger value="text">Text</TabsTrigger>
            <TabsTrigger value="image">Image</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Provider connection"
          htmlFor={providerId}
          hint="Choose where this model should run."
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
              placeholder="Choose a provider connection"
              onBlur={() => {
                const next = providerInputRef.current;
                if (next !== config.provider) {
                  applyProviderChange(next);
                }
              }}
            />
            <ComboboxContent anchor={providerAnchorRef}>
              <ComboboxEmpty>No providers found</ComboboxEmpty>
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
            ? "Add a Provider connection step first, then come back here."
            : "Matching blocks are linked automatically on the canvas."}
        </p>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Model ID"
          htmlFor={modelId}
          hint={
            isLinkedToLocal
              ? "Choose the local model Recipes should load before Run or Validate."
              : "The exact model name sent to the connection."
          }
        />
        {isLinkedToLocal && isImage ? (
          <p className="rounded-lg border border-border/60 bg-muted/10 px-3 py-2 text-xs text-muted-foreground">
            Uses the image model loaded on the Images page.
          </p>
        ) : isLinkedToLocal ? (
          <LocalRecipeModelSelector
            inputId={modelId}
            value={
              config.model.trim().toLowerCase() === "local" ? "" : config.model
            }
            ggufVariant={config.gguf_variant}
            onChange={(model, variant) =>
              onUpdate({
                model,
                // biome-ignore lint/style/useNamingConvention: api schema
                gguf_variant: variant ?? undefined,
              })
            }
          />
        ) : (
          <Input
            id={modelId}
            className="nodrag"
            placeholder={isImage ? "gpt-image-1" : "gpt-4o-mini"}
            value={config.model}
            onChange={(event) =>
              onUpdate({
                model: event.target.value,
                // biome-ignore lint/style/useNamingConvention: api schema
                gguf_variant: undefined,
              })
            }
          />
        )}
        {isLinkedToLocal && !isImage ? (
          <p className="text-xs text-muted-foreground">
            Recipes will load this model automatically. GGUF quantization is
            saved with the preset.
          </p>
        ) : null}
      </div>
      <div className="grid gap-3">
        <div className="space-y-1">
          <p className="text-sm font-semibold text-foreground">
            Default generation settings
          </p>
          <p className="text-xs text-muted-foreground">
            These defaults are reused anywhere you choose this model preset.
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {isImage ? (
            <>
              <div className="grid gap-1.5">
                <FieldLabel
                  label="Size"
                  htmlFor={sizeId}
                  hint="Width x height. Leave blank for the model's default."
                />
                <Input
                  id={sizeId}
                  className="nodrag"
                  placeholder="1024x1024"
                  value={config.image_size ?? ""}
                  onChange={(event) =>
                    updateField("image_size", event.target.value)
                  }
                />
              </div>
              <div className="grid gap-1.5">
                <FieldLabel
                  label="Steps"
                  htmlFor={stepsId}
                  hint="Denoising steps. Leave blank for the model's default."
                />
                <Input
                  id={stepsId}
                  className="nodrag"
                  value={config.image_steps ?? ""}
                  onChange={(event) =>
                    updateField("image_steps", event.target.value)
                  }
                />
              </div>
              <div className="grid gap-1.5">
                <FieldLabel
                  label="Guidance"
                  htmlFor={guidanceId}
                  hint="How closely to follow the prompt. Leave blank for the model's default."
                />
                <Input
                  id={guidanceId}
                  className="nodrag"
                  value={config.image_guidance ?? ""}
                  onChange={(event) =>
                    updateField("image_guidance", event.target.value)
                  }
                />
              </div>
            </>
          ) : (
            <>
              <div className="grid gap-1.5">
                <FieldLabel
                  label="Temperature"
                  htmlFor={tempId}
                  hint="Higher values make responses more varied."
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
                  label="Top-p"
                  htmlFor={topPId}
                  hint="Use this to limit how broad token selection can be."
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
                  label="Max tokens"
                  htmlFor={maxTokensId}
                  hint="Maximum length of the model response."
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
            </>
          )}
          <div className="grid gap-1.5">
            <FieldLabel
              label="Timeout (seconds)"
              htmlFor={timeoutId}
              hint="How long to wait before a request is treated as failed."
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
            label="Advanced request fields"
            open={optionalOpen}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-4">
          <div className="grid gap-1.5">
            <FieldLabel
              label="Advanced request fields (JSON)"
              htmlFor={extraBodyId}
              hint="Extra request fields to send with every call."
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
          <label
            htmlFor={skipHealthCheckId}
            className="flex items-center gap-2 text-xs font-semibold uppercase text-muted-foreground"
          >
            <Checkbox
              id={skipHealthCheckId}
              checked={config.skip_health_check ?? false}
              onCheckedChange={(value) =>
                updateField("skip_health_check", Boolean(value))
              }
            />
            Skip connection check
          </label>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
