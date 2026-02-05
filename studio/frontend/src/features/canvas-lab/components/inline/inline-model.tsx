import { Input } from "@/components/ui/input";
import type { ReactElement } from "react";
import type { ModelConfig, ModelProviderConfig } from "../../types";

type InlineModelPatch = Partial<ModelProviderConfig> | Partial<ModelConfig>;

type InlineModelProps = {
  config: ModelProviderConfig | ModelConfig;
  onUpdate: (patch: InlineModelPatch) => void;
};

export function InlineModel(props: InlineModelProps): ReactElement {
  if (props.config.kind === "model_provider") {
    return (
      <div className="grid grid-cols-2 gap-2">
        <Input
          className="nodrag h-7 text-xs"
          placeholder="Provider type"
          value={props.config.provider_type}
          onChange={(event) =>
            props.onUpdate({
              // biome-ignore lint/style/useNamingConvention: api schema
              provider_type: event.target.value,
            })
          }
        />
        <Input
          className="nodrag h-7 text-xs"
          placeholder="Endpoint"
          value={props.config.endpoint}
          onChange={(event) => props.onUpdate({ endpoint: event.target.value })}
        />
      </div>
    );
  }

  return (
    <div className="grid grid-cols-3 gap-2">
      <Input
        className="nodrag h-7 text-xs"
        placeholder="Provider"
        value={props.config.provider}
        onChange={(event) => props.onUpdate({ provider: event.target.value })}
      />
      <Input
        className="nodrag h-7 text-xs"
        placeholder="Model"
        value={props.config.model}
        onChange={(event) => props.onUpdate({ model: event.target.value })}
      />
      <Input
        className="nodrag h-7 text-xs"
        type="number"
        placeholder="Temp"
        value={props.config.inference_temperature ?? ""}
        onChange={(event) =>
          props.onUpdate({
            // biome-ignore lint/style/useNamingConvention: api schema
            inference_temperature: event.target.value,
          })
        }
      />
    </div>
  );
}
