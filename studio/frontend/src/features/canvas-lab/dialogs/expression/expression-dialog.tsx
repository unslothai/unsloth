import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { ReactElement } from "react";
import type { ExpressionConfig, ExpressionDtype } from "../../types";
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
        <Textarea
          id={exprId}
          className="nodrag"
          placeholder="{{ category_1 }} - {{ subcategory_1 }}"
          value={config.expr}
          onChange={(event) => updateField("expr", event.target.value)}
        />
        <p className="text-xs text-muted-foreground">
          Use Jinja2. Reference columns like {"{{ column_name }}"}.
        </p>
      </div>
    </div>
  );
}
