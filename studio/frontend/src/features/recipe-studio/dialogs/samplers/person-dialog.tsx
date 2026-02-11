import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
  const sourceId = `${config.id}-person-source`;
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
        <div className="grid gap-2">
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor={sourceId}
          >
            Source
          </label>
          <Select
            value={config.sampler_type === "person_from_faker" ? "faker" : "person"}
            onValueChange={(value) =>
              updateField(
                "sampler_type",
                value === "faker" ? "person_from_faker" : "person",
              )
            }
          >
            <SelectTrigger className="nodrag w-full" id={sourceId}>
              <SelectValue placeholder="Select source" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="person">Managed dataset</SelectItem>
              <SelectItem value="faker">Faker</SelectItem>
            </SelectContent>
          </Select>
        </div>
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
              placeholder="18-70"
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
        {config.sampler_type === "person" && (
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
        )}
      </div>
    </div>
  );
}
