import { Input } from "@/components/ui/input";
import type { ReactElement } from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";

type UniformDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function UniformDialog({
  config,
  onUpdate,
}: UniformDialogProps): ReactElement {
  const lowId = `${config.id}-uniform-low`;
  const highId = `${config.id}-uniform-high`;
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
            htmlFor={lowId}
          >
            Low
          </label>
          <Input
            id={lowId}
            type="number"
            className="nodrag"
            value={config.low ?? ""}
            onChange={(event) => onUpdate({ low: event.target.value })}
          />
        </div>
        <div className="grid gap-2">
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor={highId}
          >
            High
          </label>
          <Input
            id={highId}
            type="number"
            className="nodrag"
            value={config.high ?? ""}
            onChange={(event) => onUpdate({ high: event.target.value })}
          />
        </div>
      </div>
    </div>
  );
}
