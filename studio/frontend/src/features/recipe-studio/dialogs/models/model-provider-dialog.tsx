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
import { useI18n } from "@/features/i18n";
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
  const { t } = useI18n();
  const [optionalOpen, setOptionalOpen] = useState(false);
  const isLocal = config.is_local ?? false;
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
        label={t("recipe.modelProvider.connectionName")}
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />

      {/* Model source toggle */}
      <div className="grid gap-1.5">
        <p className="text-sm font-semibold text-foreground">
          {t("recipe.modelProvider.modelSource")}
        </p>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            className={`rounded-xl border px-4 py-3 text-left transition-colors ${
              isLocal
                ? "border-primary/40 bg-primary/5"
                : "border-border/60 bg-muted/10 hover:border-border"
            }`}
            onClick={() =>
              onUpdate({
                is_local: true,
                endpoint: "",
                api_key: "",
                api_key_env: "",
                extra_headers: "",
                extra_body: "",
              })
            }
          >
            <p className="text-sm font-semibold text-foreground">
              {t("recipe.modelProvider.localModel")}
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {t("recipe.modelProvider.localModelHint")}
            </p>
          </button>
          <button
            type="button"
            className={`rounded-xl border px-4 py-3 text-left transition-colors ${
              !isLocal
                ? "border-primary/40 bg-primary/5"
                : "border-border/60 bg-muted/10 hover:border-border"
            }`}
            onClick={() => onUpdate({ is_local: false })}
          >
            <p className="text-sm font-semibold text-foreground">
              {t("recipe.modelProvider.externalEndpoint")}
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {t("recipe.modelProvider.externalEndpointHint")}
            </p>
          </button>
        </div>
      </div>

      {isLocal ? (
        <div className="rounded-2xl border border-border/60 bg-muted/10 px-4 py-3">
          <p className="text-sm font-semibold text-foreground">
            {t("recipe.modelProvider.readyTitle")}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {t("recipe.modelProvider.readyDescription")}
          </p>
        </div>
      ) : (
        <>
          <div className="rounded-2xl border border-border/60 bg-muted/10 px-4 py-3">
            <p className="text-sm font-semibold text-foreground">
              {t("recipe.modelProvider.startTitle")}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {t("recipe.modelProvider.startDescription")}
            </p>
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label={t("recipe.modelProvider.endpoint")}
              htmlFor={endpointId}
              hint={t("recipe.modelProvider.endpointHint")}
            />
            <Input
              id={endpointId}
              className="nodrag"
              placeholder={t("recipe.modelProvider.endpointPlaceholder")}
              value={config.endpoint}
              onChange={(event) => updateField("endpoint", event.target.value)}
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label={t("recipe.modelProvider.apiKey")}
              htmlFor={apiKeyId}
              hint={t("recipe.modelProvider.apiKeyHint")}
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
                label={t("recipe.modelProvider.advanced")}
                open={optionalOpen}
              />
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 space-y-4">
              <div className="grid gap-1.5">
                <FieldLabel
                  label={t("recipe.modelProvider.apiKeyEnv")}
                  htmlFor={apiKeyEnvId}
                  hint={t("recipe.modelProvider.apiKeyEnvHint")}
                />
                <Input
                  id={apiKeyEnvId}
                  className="nodrag"
                  placeholder={t("recipe.modelProvider.apiKeyEnvPlaceholder")}
                  value={config.api_key_env ?? ""}
                  onChange={(event) =>
                    updateField("api_key_env", event.target.value)
                  }
                />
              </div>
              <div className="grid gap-1.5">
                <FieldLabel
                  label={t("recipe.modelProvider.extraHeaders")}
                  htmlFor={extraHeadersId}
                  hint={t("recipe.modelProvider.extraHeadersHint")}
                />
                <Textarea
                  id={extraHeadersId}
                  className="corner-squircle nodrag"
                  placeholder='{"X-Header": "value"}'
                  value={config.extra_headers ?? ""}
                  onChange={(event) =>
                    updateField("extra_headers", event.target.value)
                  }
                />
              </div>
              <div className="grid gap-1.5">
                <FieldLabel
                  label={t("recipe.modelProvider.extraBody")}
                  htmlFor={extraBodyId}
                  hint={t("recipe.modelProvider.extraBodyHint")}
                />
                <Textarea
                  id={extraBodyId}
                  className="corner-squircle nodrag"
                  placeholder='{"key": "value"}'
                  value={config.extra_body ?? ""}
                  onChange={(event) =>
                    updateField("extra_body", event.target.value)
                  }
                />
              </div>
            </CollapsibleContent>
          </Collapsible>
        </>
      )}
    </div>
  );
}
