import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ReactElement } from "react";
import type { ExpressionConfig, ExpressionDtype } from "../../types";
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
  return (
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
        <Input
          className="nodrag h-8 w-full text-xs"
          placeholder="{{ column_name }}"
          value={config.expr}
          onChange={(event) => onUpdate({ expr: event.target.value })}
        />
      </InlineField>
    </div>
  );
}
