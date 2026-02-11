import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type ReactElement, useState } from "react";
import type { SamplerConfig } from "../../types";
import { ChipInput } from "../../components/chip-input";
import { NameField } from "../shared/name-field";

type CategoryDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function CategoryDialog({
  config,
  onUpdate,
}: CategoryDialogProps): ReactElement {
  const [conditionDraft, setConditionDraft] = useState("");
  const conditionInputId = `${config.id}-conditional-rule`;

  const conditional = config.conditional_params ?? {};

  const handleAddCondition = () => {
    const condition = conditionDraft.trim();
    if (!condition || conditional[condition]) {
      return;
    }
    onUpdate({
      // biome-ignore lint/style/useNamingConvention: api schema
      conditional_params: {
        ...conditional,
        [condition]: {
          // biome-ignore lint/style/useNamingConvention: api schema
          sampler_type: "category",
          values: [],
          weights: [],
        },
      },
    });
    setConditionDraft("");
  };

  const removeCondition = (condition: string) => {
    const next = { ...conditional };
    delete next[condition];
    onUpdate({
      // biome-ignore lint/style/useNamingConvention: api schema
      conditional_params: Object.keys(next).length > 0 ? next : undefined,
    });
  };

  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="space-y-3">
        <div className="grid gap-2">
          <p className="text-xs font-semibold uppercase text-muted-foreground">
            Values
          </p>
          <ChipInput
            values={config.values ?? []}
            onAdd={(value) => {
              const values = [...(config.values ?? []), value];
              const weights = [...(config.weights ?? []), null];
              onUpdate({ values, weights });
            }}
            onRemove={(index) => {
              const values = [...(config.values ?? [])];
              const weights = [...(config.weights ?? [])];
              values.splice(index, 1);
              weights.splice(index, 1);
              onUpdate({ values, weights });
            }}
            placeholder="Type a value and press Enter"
          />
        </div>
        <div className="grid gap-2">
          <p className="text-xs font-semibold uppercase text-muted-foreground">
            Weights (optional)
          </p>
          <div className="grid gap-2">
            {(config.values ?? []).map((value, index) => (
              <div key={`${value}-weight`} className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground max-w-20 truncate">
                  {value}
                </span>
                <Input
                  type="number"
                  className="nodrag w-full"
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
      <div className="space-y-3 rounded-2xl border border-border/60 p-3">
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs font-semibold uppercase text-muted-foreground">
            Conditional params (category)
          </p>
          <span className="text-xs text-muted-foreground">
            {Object.keys(conditional).length} rules
          </span>
        </div>
        <div className="flex gap-2">
          <Input
            id={conditionInputId}
            className="nodrag"
            placeholder="Condition (e.g., {{ region }} == 'US')"
            value={conditionDraft}
            onChange={(event) => setConditionDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                handleAddCondition();
              }
            }}
          />
          <Button type="button" size="sm" onClick={handleAddCondition}>
            Add rule
          </Button>
        </div>
        {Object.entries(conditional).map(([condition, params]) => (
          <div
            key={condition}
            className="space-y-3 rounded-2xl border border-border/60 p-3"
          >
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-semibold text-foreground">{condition}</p>
              <Button
                type="button"
                size="xs"
                variant="ghost"
                onClick={() => removeCondition(condition)}
              >
                Remove
              </Button>
            </div>
            <ChipInput
              values={params.values ?? []}
              onAdd={(value) => {
                const values = [...(params.values ?? []), value];
                const weights = [...(params.weights ?? []), null];
                onUpdate({
                  // biome-ignore lint/style/useNamingConvention: api schema
                  conditional_params: {
                    ...conditional,
                    [condition]: { ...params, values, weights },
                  },
                });
              }}
              onRemove={(index) => {
                const values = [...(params.values ?? [])];
                const weights = [...(params.weights ?? [])];
                values.splice(index, 1);
                weights.splice(index, 1);
                onUpdate({
                  // biome-ignore lint/style/useNamingConvention: api schema
                  conditional_params: {
                    ...conditional,
                    [condition]: { ...params, values, weights },
                  },
                });
              }}
              placeholder="Type a conditional value and press Enter"
            />
            <div className="grid gap-2">
              <p className="text-xs font-semibold uppercase text-muted-foreground">
                Rule weights (optional)
              </p>
              <div className="grid gap-2">
                {(params.values ?? []).map((value, index) => (
                  <div
                    key={`${condition}-${value}-${index}-weight`}
                    className="flex items-center gap-3"
                  >
                    <span className="text-xs text-muted-foreground w-28 truncate">
                      {value}
                    </span>
                    <Input
                      type="number"
                      className="nodrag"
                      placeholder="Weight"
                      value={params.weights?.[index] ?? ""}
                      onChange={(event) => {
                        const weights = [
                          ...(params.weights ??
                            Array.from(
                              { length: (params.values ?? []).length },
                              () => null,
                            )),
                        ];
                        weights[index] = event.target.value
                          ? Number(event.target.value)
                          : null;
                        onUpdate({
                          // biome-ignore lint/style/useNamingConvention: api schema
                          conditional_params: {
                            ...conditional,
                            [condition]: { ...params, weights },
                          },
                        });
                      }}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
