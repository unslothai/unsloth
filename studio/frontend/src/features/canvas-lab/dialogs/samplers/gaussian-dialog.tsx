import { Input } from "@/components/ui/input";
import type { ReactElement } from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";

type GaussianDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function GaussianDialog({
  config,
  onUpdate,
}: GaussianDialogProps): ReactElement {
  const meanId = `${config.id}-gaussian-mean`;
  const stdId = `${config.id}-gaussian-std`;
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
            htmlFor={meanId}
          >
            Mean
          </label>
          <Input
            id={meanId}
            type="number"
            className="nodrag"
            value={config.mean ?? ""}
            onChange={(event) => onUpdate({ mean: event.target.value })}
          />
        </div>
        <div className="grid gap-2">
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor={stdId}
          >
            Std
          </label>
          <Input
            id={stdId}
            type="number"
            className="nodrag"
            value={config.std ?? ""}
            onChange={(event) => onUpdate({ std: event.target.value })}
          />
        </div>
      </div>
    </div>
  );
}
