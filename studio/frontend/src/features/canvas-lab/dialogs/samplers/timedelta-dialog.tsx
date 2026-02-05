import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type ReactElement, useMemo } from "react";
import { useCanvasLabStore } from "../../stores/canvas-lab";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";

const TIMEDELTA_UNITS: Array<"D" | "h" | "m" | "s"> = ["D", "h", "m", "s"];
const NONE_VALUE = "__none";

type TimedeltaDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function TimedeltaDialog({
  config,
  onUpdate,
}: TimedeltaDialogProps): ReactElement {
  const configs = useCanvasLabStore((state) => state.configs);
  const datetimeOptions = useMemo(
    () =>
      Object.values(configs)
        .filter(
          (item) => item.kind === "sampler" && item.sampler_type === "datetime",
        )
        .map((item) => item.name),
    [configs],
  );
  const dtMinId = `${config.id}-timedelta-min`;
  const dtMaxId = `${config.id}-timedelta-max`;
  const unitId = `${config.id}-timedelta-unit`;
  const referenceId = `${config.id}-timedelta-reference`;
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
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="grid gap-2">
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor={dtMinId}
          >
            dt_min
          </label>
          <Input
            id={dtMinId}
            type="number"
            className="nodrag"
            value={config.dt_min ?? ""}
            onChange={(event) => updateField("dt_min", event.target.value)}
          />
        </div>
        <div className="grid gap-2">
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor={dtMaxId}
          >
            dt_max
          </label>
          <Input
            id={dtMaxId}
            type="number"
            className="nodrag"
            value={config.dt_max ?? ""}
            onChange={(event) => updateField("dt_max", event.target.value)}
          />
        </div>
      </div>
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={unitId}
        >
          Unit
        </label>
        <Select
          value={config.timedelta_unit ?? "D"}
          onValueChange={(value) =>
            updateField("timedelta_unit", value as "D" | "h" | "m" | "s")
          }
        >
          <SelectTrigger className="nodrag w-full" id={unitId}>
            <SelectValue placeholder="Select unit" />
          </SelectTrigger>
          <SelectContent>
            {TIMEDELTA_UNITS.map((unit) => (
              <SelectItem key={unit} value={unit}>
                {unit}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={referenceId}
        >
          Reference datetime column
        </label>
        <Select
          value={config.reference_column_name?.trim() || NONE_VALUE}
          onValueChange={(value) =>
            updateField("reference_column_name", value === NONE_VALUE ? "" : value)
          }
        >
          <SelectTrigger className="nodrag w-full" id={referenceId}>
            <SelectValue placeholder="Select datetime column" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NONE_VALUE}>None</SelectItem>
            {datetimeOptions.map((name) => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
