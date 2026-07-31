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
import { type ReactElement, useEffect } from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";
import { FieldLabel } from "../shared/field-label";

type PersonDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function PersonDialog({
  config,
  onUpdate,
}: PersonDialogProps): ReactElement {
  const localeId = `${config.id}-person-locale`;
  const sexId = `${config.id}-person-sex`;
  const ageRangeId = `${config.id}-person-age-range`;
  const cityId = `${config.id}-person-city`;

  const updateField = <K extends keyof SamplerConfig>(
    key: K,
    value: SamplerConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<SamplerConfig>);
  };

  useEffect(() => {
    if (config.sampler_type !== "person_from_faker") {
      onUpdate({
        sampler_type: "person_from_faker",
        person_with_synthetic_personas: undefined,
      });
    }
  }, [config.sampler_type, onUpdate]);

  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-3">
        <div className="rounded-2xl border border-border/60 px-3 py-2">
          <p className="text-xs font-semibold uppercase text-muted-foreground">
            Source
          </p>
          <p className="text-sm text-foreground">Faker</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="grid gap-1.5">
            <FieldLabel
              label="Locale"
              htmlFor={localeId}
              hint="Faker locale e.g. en_US."
            />
            <Input
              id={localeId}
              className="nodrag"
              value={config.person_locale ?? ""}
              onChange={(event) =>
                updateField("person_locale", event.target.value)
              }
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Sex"
              htmlFor={sexId}
              hint="Optional demographic filter."
            />
            <Select
              value={config.person_sex?.trim() ? config.person_sex : "any"}
              onValueChange={(value) =>
                updateField("person_sex", value === "any" ? "" : value)
              }
            >
              <SelectTrigger className="nodrag w-full" id={sexId}>
                <SelectValue placeholder="Any" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="any">Any</SelectItem>
                <SelectItem value="Male">Male</SelectItem>
                <SelectItem value="Female">Female</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Age range"
              htmlFor={ageRangeId}
              hint="Range format: min-max, e.g. 18-70."
            />
            <Input
              id={ageRangeId}
              className="nodrag"
              value={config.person_age_range ?? ""}
              onChange={(event) =>
                updateField("person_age_range", event.target.value)
              }
              placeholder="18-70"
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="City"
              htmlFor={cityId}
              hint="Optional city bias for faker generation."
            />
            <Input
              id={cityId}
              className="nodrag"
              value={config.person_city ?? ""}
              onChange={(event) =>
                updateField("person_city", event.target.value)
              }
            />
          </div>
        </div>
      </div>
    </div>
  );
}
