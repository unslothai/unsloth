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
    <div className="grid grid-cols-[110px_1fr] gap-2">
      <Select
        value={config.dtype}
        onValueChange={(value) =>
          onUpdate({ dtype: value as ExpressionDtype })
        }
      >
        <SelectTrigger className="nodrag h-7 text-xs">
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
      <Input
        className="nodrag h-7 text-xs"
        placeholder="{{ column_name }}"
        value={config.expr}
        onChange={(event) => onUpdate({ expr: event.target.value })}
      />
    </div>
  );
}
