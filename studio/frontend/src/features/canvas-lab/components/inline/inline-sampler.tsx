import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ReactElement } from "react";
import type { SamplerConfig } from "../../types";

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
  return (
    <Select
      value={value ?? "none"}
      onValueChange={(next) =>
        onValueChange(next === "none" ? undefined : (next as ConvertTo))
      }
    >
      <SelectTrigger className="nodrag h-7 text-xs">
        <SelectValue placeholder="Convert" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="none">None</SelectItem>
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
  if (config.sampler_type === "uniform") {
    return (
      <div className="grid grid-cols-3 gap-2">
        <Input
          className="nodrag h-7 text-xs"
          type="number"
          placeholder="Low"
          value={config.low ?? ""}
          onChange={(event) => onUpdate({ low: event.target.value })}
        />
        <Input
          className="nodrag h-7 text-xs"
          type="number"
          placeholder="High"
          value={config.high ?? ""}
          onChange={(event) => onUpdate({ high: event.target.value })}
        />
        <ConvertToField
          value={config.convert_to}
          onValueChange={(value) =>
            onUpdate({
              // biome-ignore lint/style/useNamingConvention: api schema
              convert_to: value,
            })
          }
        />
      </div>
    );
  }

  if (config.sampler_type === "gaussian") {
    return (
      <div className="grid grid-cols-3 gap-2">
        <Input
          className="nodrag h-7 text-xs"
          type="number"
          placeholder="Mean"
          value={config.mean ?? ""}
          onChange={(event) => onUpdate({ mean: event.target.value })}
        />
        <Input
          className="nodrag h-7 text-xs"
          type="number"
          placeholder="Std"
          value={config.std ?? ""}
          onChange={(event) => onUpdate({ std: event.target.value })}
        />
        <ConvertToField
          value={config.convert_to}
          onValueChange={(value) =>
            onUpdate({
              // biome-ignore lint/style/useNamingConvention: api schema
              convert_to: value,
            })
          }
        />
      </div>
    );
  }

  if (config.sampler_type === "bernoulli") {
    return (
      <Input
        className="nodrag h-7 text-xs"
        type="number"
        min="0"
        max="1"
        step="0.01"
        placeholder="p"
        value={config.p ?? ""}
        onChange={(event) => onUpdate({ p: event.target.value })}
      />
    );
  }

  if (config.sampler_type === "uuid") {
    return (
      <Input
        className="nodrag h-7 text-xs"
        placeholder="UUID format"
        value={config.uuid_format ?? ""}
        onChange={(event) =>
          onUpdate({
            // biome-ignore lint/style/useNamingConvention: api schema
            uuid_format: event.target.value,
          })
        }
      />
    );
  }

  return null;
}
