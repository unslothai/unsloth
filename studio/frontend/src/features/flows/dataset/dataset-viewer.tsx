// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A flow `dataset` value shown with Recipe Studio's own run views: the Data tab table, the Columns tab, Raw.
// Lazy: loads the first time a dataset output is on screen.

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ExecutionColumnsTab } from "@/features/recipe-studio/components/executions/execution-columns-tab";
import { ExecutionDataTab } from "@/features/recipe-studio/components/executions/execution-data-tab";
import {
  formatCellValue,
  isExpandableCellValue,
  truncateCellValue,
} from "@/features/recipe-studio/components/executions/executions-view-helpers";
import type { RecipeExecutionRecord } from "@/features/recipe-studio/execution-types";
import { withExecutionDefaults } from "@/features/recipe-studio/executions/execution-helpers";
import { resolveImagePreview } from "@/features/recipe-studio/utils/image-preview";
import { cn } from "@/lib/utils";
import type { ColumnDef } from "@tanstack/react-table";
import { type ReactElement, useEffect, useMemo, useState } from "react";
import { datasetRows } from "../api";
import type { DatasetValue } from "../types";
import {
  type DatasetRowsPage,
  columnDtypes,
  columnNames,
  columnStats,
  datasetSummary,
  datasetTitle,
  knownTotal,
  pageCount,
} from "./dataset-value";

type Load =
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "ready"; page: DatasetRowsPage };

function useDatasetPage(value: DatasetValue, offset: number, limit: number) {
  const [load, setLoad] = useState<Load>({ state: "loading" });
  const key = JSON.stringify(value);
  // biome-ignore lint/correctness/useExhaustiveDependencies: the value is compared by its JSON
  useEffect(() => {
    let gone = false;
    setLoad((current) =>
      current.state === "ready" ? current : { state: "loading" },
    );
    datasetRows(value, offset, limit)
      .then((page) => !gone && setLoad({ state: "ready", page }))
      .catch(
        (error: unknown) =>
          !gone &&
          setLoad({
            state: "error",
            message: error instanceof Error ? error.message : String(error),
          }),
      );
    return () => {
      gone = true;
    };
  }, [key, offset, limit]);
  return load;
}

function ExpandableCell({ value, dense }: { value: unknown; dense?: boolean }) {
  const [open, setOpen] = useState(false);
  const image = resolveImagePreview(value);
  if (image?.kind === "ready")
    return (
      <img
        src={image.src}
        alt=""
        loading="lazy"
        className="h-20 w-auto max-w-56 rounded-md border border-border/60 object-contain"
      />
    );
  const text = formatCellValue(value);
  if (dense)
    return (
      <p className="line-clamp-2 min-w-24 max-w-48 break-words" title={text.slice(0, 2000)}>
        {text}
      </p>
    );
  const long = isExpandableCellValue(text);
  return (
    <div className={long ? "min-w-80 max-w-xl" : "min-w-40 max-w-md"}>
      <p className="whitespace-pre-wrap break-words">
        {long && !open ? truncateCellValue(text) : text}
      </p>
      {long && (
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="mt-1 text-[11px] text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          {open ? "Show less" : "Show all"}
        </button>
      )}
    </div>
  );
}

/** Data, Columns and Raw for one dataset, paged from the backend. */
export function DatasetViewer({
  value,
  pageSize = 20,
  dense = false,
}: {
  value: DatasetValue;
  pageSize?: number;
  dense?: boolean;
}): ReactElement {
  const [offset, setOffset] = useState(0);
  const [hidden, setHidden] = useState<string[]>([]);
  const [tab, setTab] = useState("data");
  const load = useDatasetPage(value, offset, pageSize);
  const page = load.state === "ready" ? load.page : null;
  const rows = page?.rows ?? [];
  const names = useMemo(() => columnNames(rows), [rows]);
  const dtypes = useMemo(() => columnDtypes(rows), [rows]);
  const total = knownTotal(value, page, pageSize);
  const pages = pageCount(total, page?.view_rows ?? 1000, pageSize, offset);
  const current = Math.floor(offset / pageSize) + 1;
  const columns = useMemo<ColumnDef<Record<string, unknown>>[]>(
    () =>
      names
        .filter((name) => !hidden.includes(name))
        .map((name) => ({
          accessorKey: name,
          header: () => (
            <span className="flex flex-col leading-tight">
              <span>{name}</span>
              <span className="text-[10.5px] font-normal text-muted-foreground">
                {dtypes[name]}
              </span>
            </span>
          ),
          cell: ({ getValue }) => <ExpandableCell value={getValue()} dense={dense} />,
        })),
    [names, hidden, dtypes, dense],
  );
  const execution = useMemo(
    () =>
      withExecutionDefaults({
        id: "dataset",
        status: load.state === "loading" ? "running" : "completed",
        dataset: rows,
      } as unknown as RecipeExecutionRecord),
    [rows, load.state],
  );
  return (
    <div className="flex min-w-0 flex-col gap-2" data-testid="flows-dataset-viewer">
      <p className="text-xs text-muted-foreground">
        {datasetSummary(value, total)}
        {total !== null && total > (page?.view_rows ?? Number.POSITIVE_INFINITY)
          ? ` · showing the first ${page?.view_rows.toLocaleString()}`
          : ""}
      </p>
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="border border-border/60 bg-card/40">
          <TabsTrigger value="data">Data</TabsTrigger>
          <TabsTrigger value="columns">Columns</TabsTrigger>
          <TabsTrigger value="raw">Raw</TabsTrigger>
        </TabsList>
        <TabsContent value="data">
          {load.state === "error" ? (
            <p className="mt-3 text-xs text-destructive">{load.message}</p>
          ) : (
            <ExecutionDataTab
              execution={execution}
              datasetColumnNames={names}
              hiddenDatasetColumns={hidden}
              canPageDataset={true}
              currentDatasetPage={current}
              totalPages={pages}
              tableColumns={columns}
              datasetRowsForTable={rows}
              onOpenOverview={() => setTab("raw")}
              onSetHiddenColumns={(update) => setHidden(update)}
              onPrevPage={() => setOffset(Math.max(0, offset - pageSize))}
              onNextPage={() => setOffset(offset + pageSize)}
            />
          )}
        </TabsContent>
        <TabsContent value="columns">
          <ExecutionColumnsTab analysisColumns={columnStats(rows)} />
          <p className="mt-1 text-[11px] text-muted-foreground">
            Over the {rows.length} rows on this page.
          </p>
        </TabsContent>
        <TabsContent value="raw">
          <pre className="mt-3 max-h-96 overflow-auto rounded-md bg-muted/40 p-3 text-xs">
            {JSON.stringify(
              { value, total, rows: rows.slice(0, 5) },
              null,
              2,
            )}
          </pre>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function DatasetDialog({
  value,
  open,
  onOpenChange,
}: {
  value: DatasetValue;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[85vh] flex-col gap-3 overflow-auto sm:max-w-[min(92vw,1200px)]">
        <div>
          <DialogTitle className="text-base">{datasetTitle(value)}</DialogTitle>
          <DialogDescription className="text-xs">Dataset rows</DialogDescription>
        </div>
        {open && <DatasetViewer value={value} />}
      </DialogContent>
    </Dialog>
  );
}

/** The inspector and app outputs: the full viewer, cells clamped, with Open for a bigger one. */
export function DatasetPanel({ value }: { value: DatasetValue }): ReactElement {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-end">
        <Button type="button" size="sm" variant="outline" onClick={() => setOpen(true)}>
          Open
        </Button>
      </div>
      <DatasetViewer value={value} pageSize={10} dense />
      <DatasetDialog value={value} open={open} onOpenChange={setOpen} />
    </div>
  );
}

const COMPACT_ROWS = 5;
const COMPACT_COLUMNS = 3;

/** A node's inline output: the first rows and columns, and Open for the full view. */
export function DatasetPreview({ value }: { value: DatasetValue }): ReactElement {
  const [open, setOpen] = useState(false);
  const load = useDatasetPage(value, 0, COMPACT_ROWS);
  const page = load.state === "ready" ? load.page : null;
  const rows = page?.rows ?? [];
  const names = columnNames(rows);
  const shown = names.slice(0, COMPACT_COLUMNS);
  const total = knownTotal(value, page, COMPACT_ROWS);
  return (
    <div className="nodrag flex flex-col gap-1" data-testid="flows-dataset-preview">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[10.5px] text-zinc-400">
          {datasetSummary(value, total)}
          {names.length > COMPACT_COLUMNS ? ` · ${names.length} columns` : ""}
        </span>
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="shrink-0 rounded bg-white/10 px-1.5 py-0.5 text-[10.5px] text-zinc-100 hover:bg-white/20"
        >
          Open
        </button>
      </div>
      {load.state === "error" ? (
        <p className="line-clamp-2 text-[10.5px] text-red-300">{load.message}</p>
      ) : load.state === "loading" ? (
        <p className="text-[10.5px] text-zinc-500">Loading rows…</p>
      ) : (
        <div className="overflow-hidden rounded border border-white/10">
          <table className="w-full table-fixed text-left text-[10.5px]">
            <thead className="bg-white/5 text-zinc-400">
              <tr>
                {shown.map((name) => (
                  <th key={name} className="truncate px-1.5 py-1 font-medium">
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, COMPACT_ROWS).map((row, i) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: rows have no id
                <tr key={i} className={cn("text-zinc-200", i % 2 && "bg-white/[0.03]")}>
                  {shown.map((name) => (
                    <td key={name} className="truncate px-1.5 py-0.5">
                      {formatCellValue(row[name])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <DatasetDialog value={value} open={open} onOpenChange={setOpen} />
    </div>
  );
}
