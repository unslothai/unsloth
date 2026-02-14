import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ReactElement } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { ExpressionConfig, ExpressionDtype } from "../../types";
import { getAvailableRefItems } from "../../utils/variables";
import { JinjaRefInput } from "../jinja/jinja-ref-autocomplete";
import { InlineField } from "./inline-field";

type InlineExpressionProps = {
  config: ExpressionConfig;
  onUpdate: (patch: Partial<ExpressionConfig>) => void;
};

const DTYPE_OPTIONS: ExpressionDtype[] = ["str", "int", "float", "bool"];

export function InlineExpression({
  config,
  onUpdate,
}: InlineExpressionProps): ReactElement {
  const configs = useRecipeStudioStore((state) => state.configs);
  const flowMoving = useRecipeStudioStore((state) => state.flowMoving);
  const items = getAvailableRefItems(configs, config.id);
  const vars = items.map((item) => item.ref);

  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-[130px_1fr]">
        <InlineField label="Output type">
          <Select
            value={config.dtype}
            onValueChange={(value) =>
              onUpdate({ dtype: value as ExpressionDtype })
            }
          >
            <SelectTrigger className="nodrag h-8 w-full text-xs">
              <SelectValue placeholder="dtype" />
            </SelectTrigger>
            <SelectContent>
              {DTYPE_OPTIONS.map((dtype) => (
                <SelectItem key={dtype} value={dtype}>
                  {dtype}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </InlineField>
        <InlineField label="Expression">
          <JinjaRefInput
            className="nodrag h-8 w-full text-xs"
            placeholder="{{ column_name }}"
            value={config.expr}
            items={items}
            suppress={flowMoving}
            onValueChange={(value) => onUpdate({ expr: value })}
          />
        </InlineField>
      </div>
      {vars.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground">Available references</p>
          <div className="flex flex-wrap gap-1">
            {vars.map((v) => (
              <Badge
                key={v}
                variant="secondary"
                className="corner-squircle h-4 px-1.5 font-mono text-[10px]"
              >
                {v}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
