import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { SamplerConfig } from "../../types";
import { NameField } from "../shared/name-field";

type SubcategoryDialogProps = {
  config: SamplerConfig;
  categoryOptions: SamplerConfig[];
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function SubcategoryDialog({
  config,
  categoryOptions,
  onUpdate,
}: SubcategoryDialogProps): ReactElement {
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const parentSelectId = `${config.id}-parent-category`;
  const updateField = useCallback(
    <K extends keyof SamplerConfig>(key: K, value: SamplerConfig[K]) => {
      onUpdate({ [key]: value } as Partial<SamplerConfig>);
    },
    [onUpdate],
  );
  const parent = useMemo(
    () =>
      categoryOptions.find(
        (option) => option.name === config.subcategory_parent,
      ) ?? null,
    [categoryOptions, config.subcategory_parent],
  );
  const categoryValues = parent?.values ?? [];
  const mapping = config.subcategory_mapping ?? {};

  const ensureMapping = useCallback(
    (nextParent?: SamplerConfig | null) => {
      const values = nextParent?.values ?? [];
      const nextMapping: Record<string, string[]> = {};
      for (const value of values) {
        nextMapping[value] = config.subcategory_mapping?.[value] ?? [];
      }
      const currentKeys = Object.keys(config.subcategory_mapping ?? {});
      const nextKeys = Object.keys(nextMapping);
      const changed =
        currentKeys.length !== nextKeys.length ||
        currentKeys.some((key) => !nextKeys.includes(key));
      if (changed) {
        updateField("subcategory_mapping", nextMapping);
      }
    },
    [config.subcategory_mapping, updateField],
  );

  useEffect(() => {
    if (parent) {
      ensureMapping(parent);
    }
  }, [ensureMapping, parent]);

  const addSubValue = (categoryValue: string) => {
    const draft = drafts[categoryValue]?.trim();
    if (!draft) {
      return;
    }
    const next = { ...mapping };
    const list = next[categoryValue] ? [...next[categoryValue]] : [];
    list.push(draft);
    next[categoryValue] = list;
    updateField("subcategory_mapping", next);
    setDrafts((prev) => ({ ...prev, [categoryValue]: "" }));
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
            htmlFor={parentSelectId}
          >
            Parent category column
          </label>
          <Select
            value={config.subcategory_parent ?? ""}
            onValueChange={(value) => {
              const nextParent =
                categoryOptions.find((option) => option.name === value) ?? null;
              updateField("subcategory_parent", value);
              ensureMapping(nextParent);
            }}
          >
            <SelectTrigger className="nodrag w-full" id={parentSelectId}>
              <SelectValue placeholder="Select category column" />
            </SelectTrigger>
            <SelectContent>
              {categoryOptions.map((option) => (
                <SelectItem key={option.id} value={option.name}>
                  {option.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {categoryValues.length > 0 && (
          <div className="grid gap-4">
            {categoryValues.map((value) => (
              <div
                key={value}
                className="rounded-2xl border border-border/60 p-3"
              >
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-semibold text-foreground">
                    {value}
                  </p>
                  <span className="text-xs text-muted-foreground">
                    {mapping[value]?.length ?? 0} subvalues
                  </span>
                </div>
                <div className="mt-3 flex gap-2">
                  <Input
                    className="nodrag"
                    placeholder="Add subcategory"
                    value={drafts[value] ?? ""}
                    onChange={(event) =>
                      setDrafts((prev) => ({
                        ...prev,
                        [value]: event.target.value,
                      }))
                    }
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        addSubValue(value);
                      }
                    }}
                  />
                  <Button
                    type="button"
                    size="sm"
                    onClick={() => addSubValue(value)}
                  >
                    Add
                  </Button>
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {(mapping[value] ?? []).map((item, index) => (
                    <Badge key={`${value}-${item}`} variant="secondary">
                      <span>{item}</span>
                      <button
                        type="button"
                        className="ml-2 text-xs"
                        onClick={() => {
                          const next = { ...mapping };
                          const list = [...(next[value] ?? [])];
                          list.splice(index, 1);
                          next[value] = list;
                          updateField("subcategory_mapping", next);
                        }}
                      >
                        ×
                      </button>
                    </Badge>
                  ))}
                </div>
                {(mapping[value] ?? []).length === 0 && (
                  <p className="mt-2 text-xs text-rose-500">
                    Add at least 1 subcategory.
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
