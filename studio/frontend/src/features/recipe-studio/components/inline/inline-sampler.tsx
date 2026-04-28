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
import { useI18n } from "@/features/i18n";
import type { SamplerConfig } from "../../types";
import { InlineField } from "./inline-field";

type InlineSamplerProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

type ConvertTo = "int" | "float" | "str";

function ConvertToField({
  value,
  onValueChange,
}: {
  value: SamplerConfig["convert_to"];
  onValueChange: (value: ConvertTo | undefined) => void;
}): ReactElement {
  const { t } = useI18n();
  return (
    <Select
      value={value ?? "none"}
      onValueChange={(next) =>
        onValueChange(next === "none" ? undefined : (next as ConvertTo))
      }
    >
      <SelectTrigger className="nodrag h-8 w-full text-xs">
        <SelectValue placeholder={t("recipe.sampler.common.convertTo")} />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="none">{t("recipe.sampler.common.none")}</SelectItem>
        <SelectItem value="int">int</SelectItem>
        <SelectItem value="float">float</SelectItem>
        <SelectItem value="str">str</SelectItem>
      </SelectContent>
    </Select>
  );
}

export function InlineSampler({
  config,
  onUpdate,
}: InlineSamplerProps): ReactElement | null {
  const { t } = useI18n();
  if (config.sampler_type === "uniform") {
    return (
      <div className="grid gap-3 sm:grid-cols-3">
        <InlineField label={t("recipe.sampler.uniform.low")}>
          <Input
            className="nodrag h-8 w-full text-xs"
            type="number"
            placeholder="0"
            value={config.low ?? ""}
            onChange={(event) => onUpdate({ low: event.target.value })}
          />
        </InlineField>
        <InlineField label={t("recipe.sampler.uniform.high")}>
          <Input
            className="nodrag h-8 w-full text-xs"
            type="number"
            placeholder="100"
            value={config.high ?? ""}
            onChange={(event) => onUpdate({ high: event.target.value })}
          />
        </InlineField>
        <InlineField label={t("recipe.sampler.common.convertTo")}>
          <ConvertToField
            value={config.convert_to}
            onValueChange={(value) =>
              onUpdate({
                // biome-ignore lint/style/useNamingConvention: api schema
                convert_to: value,
              })
            }
          />
        </InlineField>
      </div>
    );
  }

  if (config.sampler_type === "gaussian") {
    return (
      <div className="grid gap-3 sm:grid-cols-3">
        <InlineField label={t("recipe.sampler.gaussian.mean")}>
          <Input
            className="nodrag h-8 w-full text-xs"
            type="number"
            placeholder="0"
            value={config.mean ?? ""}
            onChange={(event) => onUpdate({ mean: event.target.value })}
          />
        </InlineField>
        <InlineField label={t("recipe.inline.sampler.stdDev")}>
          <Input
            className="nodrag h-8 w-full text-xs"
            type="number"
            placeholder="1"
            value={config.std ?? ""}
            onChange={(event) => onUpdate({ std: event.target.value })}
          />
        </InlineField>
        <InlineField label={t("recipe.sampler.common.convertTo")}>
          <ConvertToField
            value={config.convert_to}
            onValueChange={(value) =>
              onUpdate({
                // biome-ignore lint/style/useNamingConvention: api schema
                convert_to: value,
              })
            }
          />
        </InlineField>
      </div>
    );
  }

  if (config.sampler_type === "bernoulli") {
    return (
      <InlineField label={t("recipe.sampler.bernoulli.probability")}>
        <Input
          className="nodrag h-8 w-full text-xs"
          type="number"
          min="0"
          max="1"
          step="0.01"
          placeholder="0.5"
          value={config.p ?? ""}
          onChange={(event) => onUpdate({ p: event.target.value })}
        />
      </InlineField>
    );
  }

  if (config.sampler_type === "uuid") {
    return (
      <InlineField label={t("recipe.sampler.uuid.format")}>
        <Input
          className="nodrag h-8 w-full text-xs"
          placeholder="uuid4"
          value={config.uuid_format ?? ""}
          onChange={(event) =>
            onUpdate({
              // biome-ignore lint/style/useNamingConvention: api schema
              uuid_format: event.target.value,
            })
          }
        />
      </InlineField>
    );
  }

  return null;
}
