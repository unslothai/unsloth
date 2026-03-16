// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, useState } from "react";
import type { ModelProviderConfig } from "../../types";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
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
        label="Connection name"
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="rounded-2xl border border-border/60 bg-muted/10 px-4 py-3">
        <p className="text-sm font-semibold text-foreground">
          Start with the endpoint you want this model to use
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          Most connections only need an endpoint. Add an API key if that
          service requires one.
        </p>
      </div>
      <div className="grid gap-2">
        <FieldLabel
          label="Endpoint"
          htmlFor={endpointId}
          hint="Base URL for the model service or gateway."
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
          hint="Paste a key here, or use an environment variable below."
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
          <CollapsibleSectionTriggerButton
            label="Advanced request overrides"
            open={optionalOpen}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-4">
          <div className="grid gap-2">
            <FieldLabel
              label="API key environment variable"
              htmlFor={apiKeyEnvId}
              hint="Name of the environment variable that stores the key."
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
              hint="Optional headers to send with every request."
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
              hint="Optional request fields to send every time."
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
