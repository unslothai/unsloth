// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { useI18n } from "@/features/i18n";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { AnalysisColumnStat } from "./executions-view-helpers";

type ExecutionColumnsTabProps = {
  analysisColumns: AnalysisColumnStat[];
};

export function ExecutionColumnsTab({
  analysisColumns,
}: ExecutionColumnsTabProps): ReactElement {
  const { t } = useI18n();
  return (
    <div className="mt-3 rounded-xl border p-3">
      <p className="mb-2 text-sm font-semibold">{t("recipe.execution.columns.title")}</p>
      {analysisColumns.length === 0 ? (
        <p className="text-xs text-muted-foreground">{t("recipe.execution.columns.empty")}</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("recipe.execution.columns.column")}</TableHead>
              <TableHead>{t("recipe.execution.columns.type")}</TableHead>
              <TableHead>{t("recipe.execution.columns.dataType")}</TableHead>
              <TableHead>{t("recipe.execution.columns.unique")}</TableHead>
              <TableHead>{t("recipe.execution.columns.nulls")}</TableHead>
              <TableHead>{t("recipe.execution.columns.inputTokAvg")}</TableHead>
              <TableHead>{t("recipe.execution.columns.outputTokAvg")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {analysisColumns.map((column) => (
              <TableRow key={column.column_name}>
                <TableCell>{column.column_name}</TableCell>
                <TableCell>{column.column_type}</TableCell>
                <TableCell>{column.simple_dtype}</TableCell>
                <TableCell>{column.num_unique ?? "--"}</TableCell>
                <TableCell>{column.num_null ?? "--"}</TableCell>
                <TableCell>{column.input_tokens_mean ?? "--"}</TableCell>
                <TableCell>{column.output_tokens_mean ?? "--"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
