import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ReactElement } from "react";
import type { ExpressionConfig, ExpressionDtype } from "../../types";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import { getAvailableRefItems } from "../../utils/variables";
import { JinjaRefTextarea } from "../../components/jinja/jinja-ref-autocomplete";
import { AvailableVariables } from "../shared/available-variables";
import { NameField } from "../shared/name-field";

const DTYPE_OPTIONS: ExpressionDtype[] = ["str", "int", "float", "bool"];

type ExpressionDialogProps = {
  config: ExpressionConfig;
  onUpdate: (patch: Partial<ExpressionConfig>) => void;
};

export function ExpressionDialog({
  config,
  onUpdate,
}: ExpressionDialogProps): ReactElement {
  const configs = useRecipeStudioStore((state) => state.configs);
  const items = getAvailableRefItems(configs, config.id);
  const dtypeId = `${config.id}-dtype`;
  const exprId = `${config.id}-expr`;
  const updateField = <K extends keyof ExpressionConfig>(
    key: K,
    value: ExpressionConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<ExpressionConfig>);
  };
  return (
    <div className="space-y-4">
      <AvailableVariables configId={config.id} />
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={dtypeId}
        >
          Output type
        </label>
        <Select
          value={config.dtype}
          onValueChange={(value) =>
            updateField("dtype", value as ExpressionDtype)
          }
        >
          <SelectTrigger className="nodrag w-full" id={dtypeId}>
            <SelectValue placeholder="Select type" />
          </SelectTrigger>
          <SelectContent>
            {DTYPE_OPTIONS.map((dtype) => (
              <SelectItem key={dtype} value={dtype}>
                {dtype}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="grid gap-2">
        <label
          className="text-xs font-semibold uppercase text-muted-foreground"
          htmlFor={exprId}
        >
          Expression (Jinja2)
        </label>
        <JinjaRefTextarea
          id={exprId}
          className="corner-squircle nodrag"
          placeholder="{{ category_1 }} - {{ subcategory_1 }}"
          value={config.expr}
          items={items}
          onValueChange={(value) => updateField("expr", value)}
        />
        <p className="text-xs text-muted-foreground">
          Use Jinja2. Reference columns like {"{{ column_name }}"}.
        </p>
      </div>
    </div>
  );
}
