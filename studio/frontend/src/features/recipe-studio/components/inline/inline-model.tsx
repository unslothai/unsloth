// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { Input } from "@/components/ui/input";
import type { ReactElement } from "react";
import type { ModelConfig, ModelProviderConfig } from "../../types";
import { InlineField } from "./inline-field";

type InlineModelPatch = Partial<ModelProviderConfig> | Partial<ModelConfig>;

type InlineModelProps = {
  config: ModelProviderConfig | ModelConfig;
  onUpdate: (patch: InlineModelPatch) => void;
};

export function InlineModel(props: InlineModelProps): ReactElement {
  if (props.config.kind === "model_provider") {
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <InlineField label="Endpoint">
          <Input
            className="nodrag h-8 w-full text-xs"
            placeholder="https://api.example.com/v1"
            value={props.config.endpoint}
            onChange={(event) => props.onUpdate({ endpoint: event.target.value })}
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

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <InlineField label="Provider">
        <Input
          className="nodrag h-8 w-full text-xs"
          placeholder="provider alias"
          value={props.config.provider}
          onChange={(event) => props.onUpdate({ provider: event.target.value })}
        />
      </InlineField>
      <InlineField label="Model">
        <Input
          className="nodrag h-8 w-full text-xs"
          placeholder="gpt-4o-mini"
          value={props.config.model}
          onChange={(event) => props.onUpdate({ model: event.target.value })}
        />
      </InlineField>
      <InlineField label="Temperature" className="sm:col-span-2">
        <Input
          className="nodrag h-8 w-full text-xs"
          type="number"
          placeholder="0.7"
          value={props.config.inference_temperature ?? ""}
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
