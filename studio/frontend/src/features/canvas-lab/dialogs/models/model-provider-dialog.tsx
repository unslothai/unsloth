import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type { ReactElement } from "react";
import type { ModelProviderConfig } from "../../types";
import { NameField } from "../shared/name-field";

type ModelProviderDialogProps = {
  config: ModelProviderConfig;
  onUpdate: (patch: Partial<ModelProviderConfig>) => void;
};

export function ModelProviderDialog({
  config,
  onUpdate,
}: ModelProviderDialogProps): ReactElement {
  const endpointId = `${config.id}-endpoint`;
  const providerTypeId = `${config.id}-provider-type`;
  const apiKeyEnvId = `${config.id}-api-key-env`;
  const apiKeyId = `${config.id}-api-key`;
  const extraHeadersId = `${config.id}-extra-headers`;
  const extraBodyId = `${config.id}-extra-body`;
  const updateField = <K extends keyof ModelProviderConfig>(
    key: K,
    value: ModelProviderConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<ModelProviderConfig>);
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
          htmlFor={providerTypeId}
        >
          Provider type
        </label>
        <Input
          id={providerTypeId}
          className="nodrag"
          placeholder="openai"
          value={config.provider_type}
          onChange={(event) =>
            updateField("provider_type", event.target.value)
          }
        />
      </div>
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={endpointId}
        >
          Endpoint
        </label>
        <Input
          id={endpointId}
          className="nodrag"
          placeholder="https://..."
          value={config.endpoint}
          onChange={(event) => updateField("endpoint", event.target.value)}
        />
      </div>
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={apiKeyEnvId}
        >
          API key env (optional)
        </label>
        <Input
          id={apiKeyEnvId}
          className="nodrag"
          placeholder="OPENAI_API_KEY"
          value={config.api_key_env ?? ""}
          onChange={(event) => updateField("api_key_env", event.target.value)}
        />
      </div>
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={apiKeyId}
        >
          API key (optional)
        </label>
        <Input
          id={apiKeyId}
          className="nodrag"
          value={config.api_key ?? ""}
          onChange={(event) => updateField("api_key", event.target.value)}
        />
      </div>
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={extraHeadersId}
        >
          Extra headers (JSON)
        </label>
        <Textarea
          id={extraHeadersId}
          className="nodrag"
          placeholder='{"X-Header": "value"}'
          value={config.extra_headers ?? ""}
          onChange={(event) =>
            updateField("extra_headers", event.target.value)
          }
        />
      </div>
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={extraBodyId}
        >
          Extra body (JSON)
        </label>
        <Textarea
          id={extraBodyId}
          className="nodrag"
          placeholder='{"key": "value"}'
          value={config.extra_body ?? ""}
          onChange={(event) => updateField("extra_body", event.target.value)}
        />
      </div>
    </div>
  );
}
