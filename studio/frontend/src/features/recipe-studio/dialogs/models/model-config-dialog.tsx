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
import { type ReactElement, useEffect, useRef, useState } from "react";
import type { ModelConfig } from "../../types";
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
  const modelId = `${config.id}-model`;
  const providerId = `${config.id}-provider`;
  const tempId = `${config.id}-temperature`;
  const topPId = `${config.id}-top-p`;
  const maxTokensId = `${config.id}-max-tokens`;
  const providerAnchorRef = useRef<HTMLDivElement>(null);
  const [providerInput, setProviderInput] = useState(config.provider);
  useEffect(() => {
    setProviderInput(config.provider);
  }, [config.provider]);
  const updateField = <K extends keyof ModelConfig>(
    key: K,
    value: ModelConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<ModelConfig>);
  };

  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={modelId}
        >
          Model
        </label>
        <Input
          id={modelId}
          className="nodrag"
          placeholder="gpt-4o-mini"
          value={config.model}
          onChange={(event) => updateField("model", event.target.value)}
        />
      </div>
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={providerId}
        >
          Provider name
        </label>
        <div ref={providerAnchorRef}>
          <Combobox
            items={providerOptions}
            filteredItems={providerOptions}
            filter={null}
            value={config.provider || null}
            onValueChange={(value) => updateField("provider", value ?? "")}
            onInputValueChange={setProviderInput}
            itemToStringValue={(value) => value}
            autoHighlight={true}
          >
            <ComboboxInput
              id={providerId}
              className="nodrag w-full"
              placeholder="Pick provider or type name"
              onBlur={() => {
                if (providerInput !== config.provider) {
                  updateField("provider", providerInput);
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
        <label className="text-xs font-semibold uppercase text-muted-foreground">
          Inference
        </label>
        <div className="grid grid-cols-3 gap-2">
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
        </div>
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
    </div>
  );
}
