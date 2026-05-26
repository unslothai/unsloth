// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Fragment, useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import {
  getEvalRun,
  type EvalRunDetail as EvalRunDetailData,
  type EvalResultRow,
} from "../api/eval-api";
import { BreakdownTree, scoreColor } from "./breakdown-tree";

// ── Helpers ───────────────────────────────────────────────────────────

function truncate(s: string | null | undefined, n = 120): string {
  if (!s) return "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

type SortMode = "idx" | "score-asc" | "score-desc";

function nextSortMode(current: SortMode): SortMode {
  if (current === "score-asc") return "score-desc";
  if (current === "score-desc") return "idx";
  return "score-asc";
}

function ScoreSortArrow({ sort }: { sort: SortMode }) {
  if (sort === "score-asc") return <span className="ml-1 text-xs">↑</span>;
  if (sort === "score-desc") return <span className="ml-1 text-xs">↓</span>;
  return null;
}

function statusBadgeVariant(
  status: string,
): "default" | "secondary" | "destructive" | "outline" {
  if (status === "completed") return "default";
  if (status === "running") return "secondary";
  if (status === "error") return "destructive";
  // cancelled, interrupted
  return "outline";
}

// ── ExpandedDetail ────────────────────────────────────────────────────

function ExpandedDetail({ row }: { row: EvalResultRow }) {
  return (
    <div className="py-2">
      {row.error ? (
        <p className="text-sm text-red-500">{row.error}</p>
      ) : null}
      {row.breakdown ? (
        <BreakdownTree label="document" node={row.breakdown} depth={0} />
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">
              Prediction
            </p>
            <pre className="whitespace-pre-wrap break-words font-mono text-xs">
              {row.prediction_text ?? ""}
            </pre>
          </div>
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">
              Reference
            </p>
            <pre className="whitespace-pre-wrap break-words font-mono text-xs">
              {row.reference_text ?? ""}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────

export function EvalRunDetail({ runId }: { runId: string }) {
  const [detail, setDetail] = useState<EvalRunDetailData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [sort, setSort] = useState<SortMode>("score-asc");
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setDetail(null);
    setExpandedIdx(null);

    getEvalRun(runId)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch((err: unknown) => {
        if (!cancelled)
          setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [runId]);

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading run…</p>;
  }
  if (error) {
    return <p className="text-sm text-red-500">{error}</p>;
  }
  if (!detail) {
    return null;
  }

  const { run, results } = detail;

  const sorted = [...results].sort((a, b) => {
    if (sort === "idx") return a.idx - b.idx;
    const sa = a.score ?? 0;
    const sb = b.score ?? 0;
    return sort === "score-asc" ? sa - sb : sb - sa;
  });

  const numExamples = run.num_examples ?? detail.total_results;

  return (
    <div className="flex flex-col gap-4">
      {/* Aggregate header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-2">
              <span>{run.display_name ?? run.model_identifier}</span>
              <Badge variant={statusBadgeVariant(run.status)}>
                {run.status}
              </Badge>
            </div>
            <span
              className={cn(
                "text-3xl tabular-nums font-bold",
                scoreColor(run.avg_score ?? 0),
              )}
            >
              {(run.avg_score ?? 0).toFixed(4)}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-3 md:grid-cols-4">
            <div>
              <dt className="text-xs text-muted-foreground">Metric</dt>
              <dd className="font-medium">{run.metric_name}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Dataset</dt>
              <dd className="font-medium">{run.dataset_ref}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Examples</dt>
              <dd className="font-medium">{numExamples ?? "—"}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Started</dt>
              <dd className="font-medium">
                {new Date(run.started_at).toLocaleString()}
              </dd>
            </div>
            {run.ended_at ? (
              <div>
                <dt className="text-xs text-muted-foreground">Ended</dt>
                <dd className="font-medium">
                  {new Date(run.ended_at).toLocaleString()}
                </dd>
              </div>
            ) : null}
          </dl>
        </CardContent>
      </Card>

      {/* Results table */}
      <Card>
        <CardContent className="p-0">
          <Table className="w-full table-fixed">
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">Idx</TableHead>
                <TableHead
                  className="w-24 cursor-pointer select-none"
                  onClick={() => setSort(nextSortMode(sort))}
                >
                  Score
                  <ScoreSortArrow sort={sort} />
                </TableHead>
                <TableHead>Input</TableHead>
                <TableHead>Prediction</TableHead>
                <TableHead>Reference</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sorted.map((row) => {
                const isExpanded = expandedIdx === row.idx;
                return (
                  <Fragment key={row.idx}>
                    <TableRow
                      className="cursor-pointer"
                      onClick={() =>
                        setExpandedIdx(isExpanded ? null : row.idx)
                      }
                    >
                      <TableCell>{row.idx}</TableCell>
                      <TableCell>
                        {row.error ? (
                          <Badge variant="destructive">error</Badge>
                        ) : (
                          <span
                            className={cn(
                              "tabular-nums font-medium",
                              scoreColor(row.score ?? 0),
                            )}
                          >
                            {(row.score ?? 0).toFixed(3)}
                          </span>
                        )}
                      </TableCell>
                      <TableCell>
                        <div className="truncate">
                          {truncate(row.input_text)}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="truncate">
                          {truncate(row.prediction_text)}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="truncate">
                          {truncate(row.reference_text)}
                        </div>
                      </TableCell>
                    </TableRow>
                    {isExpanded ? (
                      <TableRow>
                        <TableCell colSpan={5}>
                          <ExpandedDetail row={row} />
                        </TableCell>
                      </TableRow>
                    ) : null}
                  </Fragment>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
