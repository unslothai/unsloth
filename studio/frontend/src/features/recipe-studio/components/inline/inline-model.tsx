// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import type { ReactElement } from "react";
import { LocalRecipeModelSelector } from "../../dialogs/models/local-recipe-model-selector";
import type { ModelConfig, ModelProviderConfig } from "../../types";
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
            Local model (Chat)
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

  // model_config branch: mirror the dialog path's local-aware provider sync so
  // inline edits clear stale local-only metadata without synthesizing the
  // legacy "local" placeholder.
  const localNames = props.localProviderNames ?? new Set<string>();
  const modelConfig = props.config;
  const isLinkedToLocal = localNames.has(modelConfig.provider);
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
      <InlineField label="Provider">
        <Input
          className="nodrag h-8 w-full text-xs"
          placeholder="provider alias"
          value={modelConfig.provider}
          onChange={(event) => handleProviderChange(event.target.value)}
        />
      </InlineField>
      <InlineField label="Model">
        {isLinkedToLocal ? (
          <LocalRecipeModelSelector
            compact={true}
            className="h-8 rounded-md text-xs"
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
            placeholder="gpt-4o-mini"
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
    </div>
  );
}
