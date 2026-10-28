// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { DataTable } from "@/components/ui/data-table";
import type { ColumnDef } from "@tanstack/react-table";
import { useMemo } from "react";

type Row = Record<string, unknown>;

export interface DatasetPreviewValue {
  kind: "dataset";
  preview: Row[];
  rows?: number;
  run_id?: string | null;
}

export function isDatasetPreview(value: unknown): value is DatasetPreviewValue {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: unknown }).kind === "dataset" &&
    Array.isArray((value as { preview?: unknown }).preview)
  );
}

function cell(value: unknown) {
  if (typeof value === "boolean")
    return (
      <Badge
        variant={value ? "outline" : "destructive"}
        className={
          value
            ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
            : undefined
        }
      >
        {value ? "Pass" : "Fail"}
      </Badge>
    );
  if (value === null || value === undefined || value === "")
    return <span className="text-muted-foreground/70">-</span>;
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return <span className="line-clamp-4 whitespace-pre-wrap break-words">{text}</span>;
}

/** Rows a dataset output carries inline, in the same table the dataset preview uses. */
export function DatasetRows({ value }: { value: DatasetPreviewValue }) {
  const columns = useMemo<ColumnDef<Row>[]>(() => {
    const keys: string[] = [];
    for (const row of value.preview)
      for (const key of Object.keys(row)) if (!keys.includes(key)) keys.push(key);
    return keys.map((key) => ({
      accessorKey: key,
      header: () => <span className="text-ui-11 font-semibold">{key}</span>,
      cell: ({ getValue }) => cell(getValue()),
    }));
  }, [value.preview]);
  return (
    <div className="flex flex-col gap-1">
      <div className="max-h-96 overflow-auto rounded-lg border border-border/60">
        <DataTable columns={columns} data={value.preview} />
      </div>
      <p className="text-ui-11 text-muted-foreground">
        {value.preview.length} row{value.preview.length === 1 ? "" : "s"}
        {value.run_id ? " · also on the Benchmarks page, Evals tab" : ""}
      </p>
    </div>
  );
}
