// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ReactElement } from "react";
import { LocalRecipeModelSelector } from "../../dialogs/models/local-recipe-model-selector";
import type {
  ModelConfig,
  ModelGeneration,
  ModelProviderConfig,
} from "../../types";
import { InlineField } from "./inline-field";

type InlineModelPatch = Partial<ModelProviderConfig> | Partial<ModelConfig>;

type InlineModelProps = {
  config: ModelProviderConfig | ModelConfig;
  localProviderNames?: Set<string>;
  onUpdate: (patch: InlineModelPatch) => void;
};

export function InlineModel(props: InlineModelProps): ReactElement {
  if (props.config.kind === "model_provider") {
    if (props.config.is_local) {
      return (
        <div className="flex items-center gap-2 px-1 py-0.5">
          <span className="text-xs font-medium text-muted-foreground">
            Local model (Chat and Images)
          </span>
        </div>
      );
    }
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <InlineField label="Endpoint">
          <Input
            className="nodrag h-8 w-full text-xs"
            placeholder="https://api.example.com/v1"
            value={props.config.endpoint}
            onChange={(event) =>
              props.onUpdate({ endpoint: event.target.value })
            }
          />
        </InlineField>
        <InlineField label="API key">
          <Input
            className="nodrag h-8 w-full text-xs"
            data-reload-snapshot-sensitive
            placeholder="Optional"
            value={props.config.api_key ?? ""}
            onChange={(event) =>
              props.onUpdate({
                // biome-ignore lint/style/useNamingConvention: api schema
                api_key: event.target.value,
              })
            }
          />
        </InlineField>
      </div>
    );
  }

  // Mirror the dialog path's local-aware provider sync without the legacy "local" placeholder.
  const localNames = props.localProviderNames ?? new Set<string>();
  const modelConfig = props.config;
  const isLinkedToLocal = localNames.has(modelConfig.provider);
  const isImage = modelConfig.generation === "image";
  const handleProviderChange = (nextProvider: string) => {
    const nextIsLocal = localNames.has(nextProvider);
    if (isLinkedToLocal !== nextIsLocal) {
      props.onUpdate({
        provider: nextProvider,
        model: "",
        // biome-ignore lint/style/useNamingConvention: api schema
        gguf_variant: undefined,
      });
      return;
    }
    props.onUpdate({
      provider: nextProvider,
      ...(nextIsLocal
        ? {}
        : {
            // biome-ignore lint/style/useNamingConvention: api schema
            gguf_variant: undefined,
          }),
    });
  };

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <InlineField label="Generates" className="sm:col-span-2">
        <Select
          value={isImage ? "image" : "text"}
          onValueChange={(value) =>
            props.onUpdate({ generation: value as ModelGeneration })
          }
        >
          <SelectTrigger className="nodrag h-8 w-full text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="text">Text</SelectItem>
            <SelectItem value="image">Image</SelectItem>
          </SelectContent>
        </Select>
      </InlineField>
      <InlineField label="Provider">
        <Input
          className="nodrag h-8 w-full text-xs"
          placeholder="provider alias"
          value={modelConfig.provider}
          onChange={(event) => handleProviderChange(event.target.value)}
        />
      </InlineField>
      <InlineField label="Model">
        {isLinkedToLocal && isImage ? (
          <p className="flex h-8 items-center px-1 text-xs text-muted-foreground">
            Loaded image model
          </p>
        ) : isLinkedToLocal ? (
          <LocalRecipeModelSelector
            compact={true}
            className="h-8 rounded-full text-xs"
            value={
              modelConfig.model.trim().toLowerCase() === "local"
                ? ""
                : modelConfig.model
            }
            ggufVariant={modelConfig.gguf_variant}
            onChange={(model, variant) =>
              props.onUpdate({
                model,
                // biome-ignore lint/style/useNamingConvention: api schema
                gguf_variant: variant ?? undefined,
              })
            }
          />
        ) : (
          <Input
            className="nodrag h-8 w-full text-xs"
            placeholder={isImage ? "gpt-image-1" : "gpt-4o-mini"}
            value={modelConfig.model}
            onChange={(event) =>
              props.onUpdate({
                model: event.target.value,
                // biome-ignore lint/style/useNamingConvention: api schema
                gguf_variant: undefined,
              })
            }
          />
        )}
      </InlineField>
      {isImage ? (
        <InlineField label="Size" className="sm:col-span-2">
          <Input
            className="nodrag h-8 w-full text-xs"
            placeholder="1024x1024"
            value={modelConfig.image_size ?? ""}
            onChange={(event) =>
              props.onUpdate({
                // biome-ignore lint/style/useNamingConvention: api schema
                image_size: event.target.value,
              })
            }
          />
        </InlineField>
      ) : (
        <InlineField label="Temperature" className="sm:col-span-2">
          <Input
            className="nodrag h-8 w-full text-xs"
            type="number"
            placeholder="0.7"
            value={modelConfig.inference_temperature ?? ""}
            onChange={(event) =>
              props.onUpdate({
                // biome-ignore lint/style/useNamingConvention: api schema
                inference_temperature: event.target.value,
              })
            }
          />
        </InlineField>
      )}
    </div>
  );
}
