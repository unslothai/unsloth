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
import { type ReactElement, useRef, useState } from "react";
import type { ModelConfig } from "../../types";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type ModelConfigDialogProps = {
  config: ModelConfig;
  providerOptions: string[];
  onUpdate: (patch: Partial<ModelConfig>) => void;
};

export function ModelConfigDialog({
  config,
  providerOptions,
  onUpdate,
}: ModelConfigDialogProps): ReactElement {
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
  const lastProviderRef = useRef(config.provider);
  if (lastProviderRef.current !== config.provider) {
    lastProviderRef.current = config.provider;
    providerInputRef.current = config.provider;
  }
  const updateField = <K extends keyof ModelConfig>(
    key: K,
    value: ModelConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<ModelConfig>);
  };

  return (
    <div className="space-y-4">
      <NameField
        label="Model alias"
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-2">
        <FieldLabel
          label="Model"
          htmlFor={modelId}
          hint="Exact model id string sent to provider."
        />
        <Input
          id={modelId}
          className="nodrag"
          placeholder="gpt-4o-mini"
          value={config.model}
          onChange={(event) => updateField("model", event.target.value)}
        />
      </div>
      <div className="grid gap-2">
        <FieldLabel
          label="Provider name"
          htmlFor={providerId}
          hint="Must match a Model Provider block name."
        />
        <div ref={providerAnchorRef}>
          <Combobox
            items={providerOptions}
            filteredItems={providerOptions}
            filter={null}
            value={config.provider || null}
            onValueChange={(value) => updateField("provider", value ?? "")}
            onInputValueChange={(value) => {
              providerInputRef.current = value;
            }}
            itemToStringValue={(value) => value}
            autoHighlight={true}
          >
            <ComboboxInput
              id={providerId}
              className="nodrag w-full"
              placeholder="Pick provider or type name"
              onBlur={() => {
                const next = providerInputRef.current;
                if (next !== config.provider) {
                  updateField("provider", next);
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
          Pick provider name from list. Matching node link becomes semantic.
        </p>
      </div>
      <div className="grid gap-2">
        <FieldLabel
          label="Inference"
          hint="Runtime generation params for this model alias."
        />
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Input
            id={tempId}
            className="nodrag"
            placeholder="Temp"
            value={config.inference_temperature ?? ""}
            onChange={(event) =>
              updateField("inference_temperature", event.target.value)
            }
          />
          <Input
            id={topPId}
            className="nodrag"
            placeholder="Top_p"
            value={config.inference_top_p ?? ""}
            onChange={(event) =>
              updateField("inference_top_p", event.target.value)
            }
          />
          <Input
            id={maxTokensId}
            className="nodrag"
            placeholder="Max tokens"
            value={config.inference_max_tokens ?? ""}
            onChange={(event) =>
              updateField("inference_max_tokens", event.target.value)
            }
          />
          <Input
            id={timeoutId}
            className="nodrag"
            placeholder="Timeout (sec)"
            value={config.inference_timeout ?? ""}
            onChange={(event) =>
              updateField("inference_timeout", event.target.value)
            }
          />
        </div>
      </div>
      <Collapsible open={optionalOpen} onOpenChange={setOptionalOpen}>
        <CollapsibleTrigger asChild={true}>
          <button
            type="button"
            className="flex w-full items-center justify-between text-left text-xs text-muted-foreground"
          >
            <span className="font-semibold uppercase">Optional</span>
            <span>{optionalOpen ? "Hide" : "Show"}</span>
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-4">
          <div className="grid gap-2">
            <FieldLabel
              label="Inference extra body (JSON)"
              htmlFor={extraBodyId}
              hint="Optional request fields merged into inference parameters."
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
            Skip health check
          </label>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
