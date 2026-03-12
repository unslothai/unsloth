// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
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
import { findInvalidJinjaReferences } from "../../utils/refs";
import { getAvailableVariableEntries } from "../../utils/variables";
import { AvailableReferencesInline } from "../shared/available-references-inline";
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
  const vars = getAvailableVariableEntries(configs, config.id);
  const invalidRefs = findInvalidJinjaReferences(
    config.expr,
    vars.map((entry) => entry.name),
  );

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
          <Input
            className="nodrag h-8 w-full text-xs"
            aria-invalid={invalidRefs.length > 0}
            placeholder="{{ column_name }}"
            value={config.expr}
            onChange={(event) => onUpdate({ expr: event.target.value })}
          />
        </InlineField>
      </div>
      <AvailableReferencesInline entries={vars} />
    </div>
  );
}
