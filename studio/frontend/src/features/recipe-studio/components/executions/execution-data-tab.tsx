// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { Button } from "@/components/ui/button";
import { DataTable } from "@/components/ui/data-table";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { isExecutionInProgress } from "../../executions/execution-helpers";
import type { RecipeExecutionRecord } from "../../execution-types";
import { hasExpandableTextCell } from "./executions-view-helpers";

type ExecutionDataTabProps = {
  execution: RecipeExecutionRecord;
  datasetColumnNames: string[];
  hiddenDatasetColumns: string[];
  canPageDataset: boolean;
  currentDatasetPage: number;
  totalPages: number;
  tableColumns: ColumnDef<Record<string, unknown>>[];
  datasetRowsForTable: Record<string, unknown>[];
  visibleDatasetColumnNames: string[];
  expandedDatasetRows: Record<string, boolean>;
  selectedExecutionIdSafe: string | null;
  onSetHiddenColumns: (updater: (current: string[]) => string[]) => void;
  onPrevPage: () => void;
  onNextPage: () => void;
  onToggleRowExpanded: (rowId: string) => void;
};

export function ExecutionDataTab({
  execution,
  datasetColumnNames,
  hiddenDatasetColumns,
  canPageDataset,
  currentDatasetPage,
  totalPages,
  tableColumns,
  datasetRowsForTable,
  visibleDatasetColumnNames,
  expandedDatasetRows,
  selectedExecutionIdSafe,
  onSetHiddenColumns,
  onPrevPage,
  onNextPage,
  onToggleRowExpanded,
}: ExecutionDataTabProps): ReactElement {
  return (
    <div className="mt-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-semibold">Dataset sample</p>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {datasetColumnNames.length > 0 && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button type="button" size="sm" variant="outline">
                  Columns
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel>Visible columns</DropdownMenuLabel>
                {datasetColumnNames.map((columnName) => (
                  <DropdownMenuCheckboxItem
                    key={columnName}
                    checked={!hiddenDatasetColumns.includes(columnName)}
                    onSelect={(event) => {
                      event.preventDefault();
                    }}
                    onCheckedChange={(checked) => {
                      onSetHiddenColumns((currentColumns) => {
                        if (checked) {
                          return currentColumns.filter((name) => name !== columnName);
                        }
                        return [...currentColumns, columnName];
                      });
                    }}
                  >
                    {columnName}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          {canPageDataset && (
            <>
              <span>
                Page {currentDatasetPage}/{totalPages}
              </span>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={
                  isExecutionInProgress(execution.status) || currentDatasetPage <= 1
                }
                onClick={onPrevPage}
              >
                Prev
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={
                  isExecutionInProgress(execution.status) ||
                  currentDatasetPage >= totalPages
                }
                onClick={onNextPage}
              >
                Next
              </Button>
            </>
          )}
        </div>
      </div>
      {execution.dataset.length === 0 ? (
        isExecutionInProgress(execution.status) ? (
          <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
            <Spinner className="size-5" />
            <div className="space-y-1">
              <p className="text-sm font-medium text-muted-foreground">
                Generating data…
              </p>
              <p className="text-xs text-muted-foreground">
                Check the Overview tab for live terminal logs.
              </p>
            </div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">No rows returned.</p>
        )
      ) : tableColumns.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          All columns hidden. Use Columns to show at least one.
        </p>
      ) : (
        <div className="max-h-[55vh] overflow-auto">
          <DataTable
            columns={tableColumns}
            data={datasetRowsForTable}
            getRowClassName={(row, _rowIndex, rowId) => {
              const canExpand = hasExpandableTextCell(row, visibleDatasetColumnNames);
              if (!canExpand) {
                return undefined;
              }
              return cn(
                "cursor-pointer",
                expandedDatasetRows[rowId] ? "bg-primary/[0.05]" : "hover:bg-primary/[0.06]",
              );
            }}
            onRowClick={(row, _rowIndex, rowId) => {
              const canExpand = hasExpandableTextCell(row, visibleDatasetColumnNames);
              if (!canExpand || !selectedExecutionIdSafe) {
                return;
              }
              onToggleRowExpanded(rowId);
            }}
          />
        </div>
      )}
    </div>
  );
}
