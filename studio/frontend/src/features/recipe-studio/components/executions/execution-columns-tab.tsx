// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
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
  return (
    <div className="mt-3 rounded-xl border p-3">
      <p className="mb-2 text-sm font-semibold">Column statistics</p>
      {analysisColumns.length === 0 ? (
        <p className="text-xs text-muted-foreground">No column statistics yet.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Column</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Data type</TableHead>
              <TableHead>Unique</TableHead>
              <TableHead>Nulls</TableHead>
              <TableHead>Input tok avg</TableHead>
              <TableHead>Output tok avg</TableHead>
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
