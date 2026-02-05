import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type ReactElement, useState } from "react";
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
  const [conditionDraft, setConditionDraft] = useState("");
  const [conditionalValueDrafts, setConditionalValueDrafts] = useState<
    Record<string, string>
  >({});
  const valuesInputId = `${config.id}-values`;
  const conditionInputId = `${config.id}-conditional-rule`;

  const conditional = config.conditional_params ?? {};

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

  const addConditionalValue = (condition: string) => {
    const draft = conditionalValueDrafts[condition]?.trim();
    if (!draft) {
      return;
    }
    const current = conditional[condition] ?? {
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "category" as const,
      values: [],
      weights: [],
    };
    const values = [...(current.values ?? []), draft];
    const weights = [...(current.weights ?? []), null];
    onUpdate({
      // biome-ignore lint/style/useNamingConvention: api schema
      conditional_params: {
        ...conditional,
        [condition]: { ...current, values, weights },
      },
    });
    setConditionalValueDrafts((prev) => ({ ...prev, [condition]: "" }));
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
            <div className="flex gap-2">
              <Input
                className="nodrag"
                placeholder="Add conditional value"
                value={conditionalValueDrafts[condition] ?? ""}
                onChange={(event) =>
                  setConditionalValueDrafts((prev) => ({
                    ...prev,
                    [condition]: event.target.value,
                  }))
                }
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    addConditionalValue(condition);
                  }
                }}
              />
              <Button
                type="button"
                size="sm"
                onClick={() => addConditionalValue(condition)}
              >
                Add
              </Button>
            </div>
            <div className="flex flex-wrap gap-2">
              {(params.values ?? []).map((value, index) => (
                <Badge
                  key={`${condition}-${value}-${index}`}
                  variant="secondary"
                >
                  <span>{value}</span>
                  <button
                    type="button"
                    className="ml-2 text-xs"
                    onClick={() => {
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
                  >
                    ×
                  </button>
                </Badge>
              ))}
            </div>
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
