// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ColumnDef,
  type SortingState,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useState } from "react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  className?: string;
  onRowClick?: (row: TData, rowIndex: number, rowId: string) => void;
  getRowClassName?: (
    row: TData,
    rowIndex: number,
    rowId: string,
  ) => string | undefined;
}

export function DataTable<TData, TValue>({
  columns,
  data,
  className,
  onRowClick,
  getRowClassName,
}: DataTableProps<TData, TValue>) {
  const [sorting, setSorting] = useState<SortingState>([]);

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    state: { sorting },
  });

  return (
    <div className={cn("w-full", className)}>
      <Table>
        <TableHeader className="sticky top-0 z-10">
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow
              key={headerGroup.id}
              className="bg-muted/60 hover:bg-muted/60 border-b border-border/60"
            >
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  className="border-r border-border/40 last:border-r-0 h-11 px-4 text-xs"
                  style={{
                    width:
                      header.getSize() !== 150 ? header.getSize() : undefined,
                  }}
                >
                  {header.isPlaceholder
                    ? null
                    : flexRender(
                        header.column.columnDef.header,
                        header.getContext(),
                      )}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.length ? (
            table.getRowModel().rows.map((row, idx) => (
              <TableRow
                key={row.id}
                data-state={row.getIsSelected() ? "selected" : undefined}
                className={cn(
                  "transition-colors border-b border-border/30",
                  idx % 2 === 0
                    ? "bg-background"
                    : "bg-muted/20",
                  "hover:bg-primary/[0.03]",
                  getRowClassName?.(row.original, idx, row.id),
                )}
                onClick={() => onRowClick?.(row.original, idx, row.id)}
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell
                    key={cell.id}
                    className="border-r border-border/20 last:border-r-0 text-[13px] py-3 px-4 align-top whitespace-normal"
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell
                colSpan={columns.length}
                className="h-32 text-center text-muted-foreground text-sm"
              >
                No results.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
