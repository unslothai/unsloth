import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type ReactElement, useEffect, useState } from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";

type CategoryDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function CategoryDialog({
  config,
  onUpdate,
}: CategoryDialogProps): ReactElement {
  const [valueDraft, setValueDraft] = useState("");
  const valuesInputId = `${config.id}-values`;

  useEffect(() => {
    if (config.id) {
      setValueDraft("");
    }
  }, [config.id]);

  const handleAddValue = () => {
    const nextValue = valueDraft.trim();
    if (!nextValue) {
      return;
    }
    const values = config.values ? [...config.values] : [];
    const weights = config.weights ? [...config.weights] : [];
    values.push(nextValue);
    weights.push(null);
    onUpdate({ values, weights });
    setValueDraft("");
  };

  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="space-y-3">
        <div className="grid gap-2">
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor={valuesInputId}
          >
            Values
          </label>
          <div className="flex gap-2">
            <Input
              id={valuesInputId}
              className="nodrag"
              placeholder="Add a value"
              value={valueDraft}
              onChange={(event) => setValueDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  handleAddValue();
                }
              }}
            />
            <Button type="button" size="sm" onClick={handleAddValue}>
              Add
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            {(config.values ?? []).map((value, index) => (
              <Badge key={value} variant="secondary">
                <span>{value}</span>
                <button
                  type="button"
                  className="ml-2 text-xs"
                  onClick={() => {
                    const values = [...(config.values ?? [])];
                    const weights = [...(config.weights ?? [])];
                    values.splice(index, 1);
                    weights.splice(index, 1);
                    onUpdate({ values, weights });
                  }}
                >
                  ×
                </button>
              </Badge>
            ))}
          </div>
        </div>
        <div className="grid gap-2">
          <p className="text-xs font-semibold uppercase text-muted-foreground">
            Weights (optional)
          </p>
          <div className="grid gap-2">
            {(config.values ?? []).map((value, index) => (
              <div key={`${value}-weight`} className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground w-20 truncate">
                  {value}
                </span>
                <Input
                  type="number"
                  className="nodrag"
                  placeholder="Weight"
                  value={config.weights?.[index] ?? ""}
                  onChange={(event) => {
                    const weights = [...(config.weights ?? [])];
                    weights[index] = event.target.value
                      ? Number(event.target.value)
                      : null;
                    onUpdate({ weights });
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
