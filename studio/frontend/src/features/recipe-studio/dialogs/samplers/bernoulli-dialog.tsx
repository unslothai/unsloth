import { Input } from "@/components/ui/input";
import type { ReactElement } from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";

type BernoulliDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function BernoulliDialog({
  config,
  onUpdate,
}: BernoulliDialogProps): ReactElement {
  const pId = `${config.id}-bernoulli-p`;
  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={pId}
        >
          Probability (p)
        </label>
        <Input
          id={pId}
          type="number"
          min="0"
          max="1"
          step="0.01"
          className="nodrag"
          value={config.p ?? ""}
          onChange={(event) => onUpdate({ p: event.target.value })}
        />
      </div>
    </div>
  );
}
