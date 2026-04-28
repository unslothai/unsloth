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
import { NameField } from "../shared/name-field";
import { FieldLabel } from "../shared/field-label";

type GaussianDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function GaussianDialog({
  config,
  onUpdate,
}: GaussianDialogProps): ReactElement {
  const { t } = useI18n();
  const meanId = `${config.id}-gaussian-mean`;
  const stdId = `${config.id}-gaussian-std`;
  const convertId = `${config.id}-gaussian-convert`;
  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="grid gap-1.5">
          <FieldLabel
            label={t("recipe.sampler.gaussian.mean")}
            htmlFor={meanId}
            hint={t("recipe.sampler.gaussian.meanHint")}
          />
          <Input
            id={meanId}
            type="number"
            className="nodrag"
            value={config.mean ?? ""}
            onChange={(event) => onUpdate({ mean: event.target.value })}
          />
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label={t("recipe.sampler.gaussian.std")}
            htmlFor={stdId}
            hint={t("recipe.sampler.gaussian.stdHint")}
          />
          <Input
            id={stdId}
            type="number"
            className="nodrag"
            value={config.std ?? ""}
            onChange={(event) => onUpdate({ std: event.target.value })}
          />
        </div>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label={t("recipe.sampler.common.convertTo")}
          htmlFor={convertId}
          hint={t("recipe.sampler.common.convertHint")}
        />
        <Select
          value={config.convert_to ?? "none"}
          onValueChange={(value) =>
            onUpdate({
              // biome-ignore lint/style/useNamingConvention: api schema
              convert_to: value === "none" ? undefined : (value as "int" | "float" | "str"),
            })
          }
        >
          <SelectTrigger className="nodrag w-full" id={convertId}>
            <SelectValue placeholder={t("recipe.sampler.common.noConversion")} />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">{t("recipe.sampler.common.none")}</SelectItem>
            <SelectItem value="int">int</SelectItem>
            <SelectItem value="float">float</SelectItem>
            <SelectItem value="str">str</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
