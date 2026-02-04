import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import type { ReactElement } from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";

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
  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="grid gap-2">
            <label
              className="text-xs font-semibold uppercase text-muted-foreground"
              htmlFor={localeId}
            >
              Locale
            </label>
            <Input
              id={localeId}
              className="nodrag"
              value={config.person_locale ?? ""}
              onChange={(event) =>
                updateField("person_locale", event.target.value)
              }
            />
          </div>
          <div className="grid gap-2">
            <label
              className="text-xs font-semibold uppercase text-muted-foreground"
              htmlFor={sexId}
            >
              Sex
            </label>
            <Input
              id={sexId}
              className="nodrag"
              value={config.person_sex ?? ""}
              onChange={(event) =>
                updateField("person_sex", event.target.value)
              }
            />
          </div>
          <div className="grid gap-2">
            <label
              className="text-xs font-semibold uppercase text-muted-foreground"
              htmlFor={ageRangeId}
            >
              Age range
            </label>
            <Input
              id={ageRangeId}
              className="nodrag"
              value={config.person_age_range ?? ""}
              onChange={(event) =>
                updateField("person_age_range", event.target.value)
              }
            />
          </div>
          <div className="grid gap-2">
            <label
              className="text-xs font-semibold uppercase text-muted-foreground"
              htmlFor={cityId}
            >
              City
            </label>
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
        <div className="flex items-center justify-between gap-3 rounded-2xl border border-border/60 px-3 py-2">
          <div>
            <p className="text-sm font-semibold">Synthetic personas</p>
            <p className="text-xs text-muted-foreground">
              Generate persona profiles.
            </p>
          </div>
          <Switch
            checked={config.person_with_synthetic_personas ?? false}
            onCheckedChange={(value) =>
              updateField("person_with_synthetic_personas", value)
            }
          />
        </div>
        <div className="flex items-center justify-between gap-3 rounded-2xl border border-border/60 px-3 py-2">
          <div>
            <p className="text-sm font-semibold">Sample dataset</p>
            <p className="text-xs text-muted-foreground">
              Use dataset when available.
            </p>
          </div>
          <Switch
            checked={config.person_sample_dataset_when_available ?? false}
            onCheckedChange={(value) =>
              updateField("person_sample_dataset_when_available", value)
            }
          />
        </div>
      </div>
    </div>
  );
}
