import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import type { ModelProviderConfig } from "../../types";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type ModelProviderDialogProps = {
  config: ModelProviderConfig;
  onUpdate: (patch: Partial<ModelProviderConfig>) => void;
};

export function ModelProviderDialog({
  config,
  onUpdate,
}: ModelProviderDialogProps): ReactElement {
  const [optionalOpen, setOptionalOpen] = useState(false);
  const endpointId = `${config.id}-endpoint`;
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
        <FieldLabel
          label="Endpoint"
          htmlFor={endpointId}
          hint="Base API URL used for model requests."
        />
        <Input
          id={endpointId}
          className="nodrag"
          placeholder="https://..."
          value={config.endpoint}
          onChange={(event) => updateField("endpoint", event.target.value)}
        />
      </div>
      <div className="grid gap-2">
        <FieldLabel
          label="API key (optional)"
          htmlFor={apiKeyId}
          hint="Inline key. prefer env var for safer configs."
        />
        <Input
          id={apiKeyId}
          className="nodrag"
          value={config.api_key ?? ""}
          onChange={(event) => updateField("api_key", event.target.value)}
        />
      </div>
      <Collapsible open={optionalOpen} onOpenChange={setOptionalOpen}>
        <CollapsibleTrigger asChild={true}>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="w-full justify-between"
          >
            Optional
            <HugeiconsIcon
              icon={ArrowDown01Icon}
              strokeWidth={2}
              className={
                optionalOpen ? "rotate-180 transition-transform" : "transition-transform"
              }
            />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-4">
          <div className="grid gap-2">
            <FieldLabel
              label="API key env (optional)"
              htmlFor={apiKeyEnvId}
              hint="Env var name to read secret key from runtime."
            />
            <Input
              id={apiKeyEnvId}
              className="nodrag"
              placeholder="OPENAI_API_KEY"
              value={config.api_key_env ?? ""}
              onChange={(event) => updateField("api_key_env", event.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <FieldLabel
              label="Extra headers (JSON)"
              htmlFor={extraHeadersId}
              hint="Optional request headers merged into every call."
            />
            <Textarea
              id={extraHeadersId}
              className="corner-squircle nodrag"
              placeholder='{"X-Header": "value"}'
              value={config.extra_headers ?? ""}
              onChange={(event) => updateField("extra_headers", event.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <FieldLabel
              label="Extra body (JSON)"
              htmlFor={extraBodyId}
              hint="Optional payload fields merged into requests."
            />
            <Textarea
              id={extraBodyId}
              className="corner-squircle nodrag"
              placeholder='{"key": "value"}'
              value={config.extra_body ?? ""}
              onChange={(event) => updateField("extra_body", event.target.value)}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
