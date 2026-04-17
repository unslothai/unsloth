// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { ReactElement } from "react";
import { useMemo } from "react";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { ExpressionConfig, ExpressionDtype } from "../../types";
import { findInvalidJinjaReferences } from "../../utils/refs";
import { getAvailableVariables } from "../../utils/variables";
import { AvailableVariables } from "../shared/available-variables";
import { FieldLabel } from "../shared/field-label";
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
  const dtypeId = `${config.id}-dtype`;
  const exprId = `${config.id}-expr`;
  const validReferences = useMemo(
    () => getAvailableVariables(configs, config.id),
    [configs, config.id],
  );
  const invalidExprRefs = useMemo(
    () => findInvalidJinjaReferences(config.expr, validReferences),
    [config.expr, validReferences],
  );
  const invalidExprText = invalidExprRefs
    .slice(0, 3)
    .map((ref) => `{{ ${ref} }}`)
    .join(", ");
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
      <div className="grid gap-1.5">
        <FieldLabel
          label="输出类型"
          htmlFor={dtypeId}
          hint="选择该公式在最终数据集中的保存类型。"
        />
        <Select
          value={config.dtype}
          onValueChange={(value) =>
            updateField("dtype", value as ExpressionDtype)
          }
        >
          <SelectTrigger className="nodrag w-full" id={dtypeId}>
            <SelectValue placeholder="选择类型" />
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
      <div className="grid gap-1.5">
        <FieldLabel
          label="公式"
          htmlFor={exprId}
          hint="使用其他字段组合生成当前字段。"
        />
        <Textarea
          id={exprId}
          className="corner-squircle nodrag"
          aria-invalid={invalidExprRefs.length > 0}
          placeholder="{{ category_1 }} - {{ subcategory_1 }}"
          value={config.expr}
          onChange={(event) => updateField("expr", event.target.value)}
        />
        {invalidExprRefs.length > 0 && (
          <p className="text-xs text-destructive">
            未知字段：{invalidExprText}
            {invalidExprRefs.length > 3
              ? ` 另有 ${invalidExprRefs.length - 3} 个`
              : ""}
          </p>
        )}
        <p className="text-xs text-muted-foreground">
          可通过 {"{{ field_name }}"} 引用其他字段。
        </p>
      </div>
    </div>
  );
}
