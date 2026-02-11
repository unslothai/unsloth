import { Input } from "@/components/ui/input";
import type { ReactElement } from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";

type UuidDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function UuidDialog({
  config,
  onUpdate,
}: UuidDialogProps): ReactElement {
  const uuidId = `${config.id}-uuid-format`;
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
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={uuidId}
        >
          UUID format (optional)
        </label>
        <Input
          id={uuidId}
          className="nodrag"
          value={config.uuid_format ?? ""}
          onChange={(event) => updateField("uuid_format", event.target.value)}
        />
      </div>
    </div>
  );
}
