// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";
import { cancelEval } from "../api/eval-api";
import { scoreColor } from "./breakdown-tree";
import { EvalRunDetail } from "./eval-run-detail";
import { useEvalRuntimeStore } from "../stores/eval-runtime-store";

// ── Helpers ───────────────────────────────────────────────────────────────

function logColor(level: string): string {
  if (level === "error") return "text-red-500";
  if (level === "warning" || level === "warn") return "text-amber-500";
  return "text-foreground/80";
}

function formatDuration(totalSec: number): string {
  const s = Math.max(0, Math.round(totalSec));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${rem}s`;
}

// ── Component ─────────────────────────────────────────────────────────────

export function LiveEvalView() {
  const currentRunId = useEvalRuntimeStore((s) => s.currentRunId);
  const status = useEvalRuntimeStore((s) => s.status);
  const done = useEvalRuntimeStore((s) => s.done);
  const total = useEvalRuntimeStore((s) => s.total);
  const avgScore = useEvalRuntimeStore((s) => s.avgScore);
  const startedAtMs = useEvalRuntimeStore((s) => s.startedAtMs);
  const liveResults = useEvalRuntimeStore((s) => s.liveResults);
  const isEvalRunning = useEvalRuntimeStore((s) => s.isEvalRunning);
  const logs = useEvalRuntimeStore((s) => s.logs);

  const logsEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = logsEndRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs.length]);

  const etaSec =
    startedAtMs && done > 0 && total > done
      ? ((Date.now() - startedAtMs) / done) * (total - done) / 1000
      : null;

  const isTerminal = status !== "running" && status !== "idle";

  async function handleCancel() {
    if (!currentRunId) return;
    try {
      await cancelEval(currentRunId);
    } catch (err) {
      toast.error("Failed to cancel eval", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  if (!currentRunId) {
    return (
      <p className="text-sm text-muted-foreground">
        No active eval. Configure one to get started.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Header card — progress summary */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-4">
            <div className="flex flex-col gap-0.5">
              <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Avg score
              </CardTitle>
              <span className={cn("text-3xl tabular-nums font-bold", scoreColor(avgScore))}>
                {avgScore.toFixed(4)}
              </span>
            </div>
            {isEvalRunning && (
              <Button variant="destructive" size="sm" onClick={handleCancel}>
                Cancel
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          <Progress value={total ? (done / total) * 100 : 0} />
          <p className="text-xs text-muted-foreground">
            {done} / {total || "?"} examples
            {etaSec != null && isEvalRunning && (
              <> · ~{formatDuration(etaSec)} left</>
            )}
            {" · "}{status}
          </p>
        </CardContent>
      </Card>

      {/* Logs card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Logs</CardTitle>
        </CardHeader>
        <CardContent>
          <div
            ref={logsEndRef}
            className="max-h-72 overflow-auto rounded-md bg-muted/40 p-2 font-mono text-xs leading-relaxed"
          >
            {logs.length === 0 ? (
              <span className="text-muted-foreground">Waiting for logs…</span>
            ) : (
              logs.map((e) => (
                <div
                  key={e.seq}
                  className={cn("whitespace-pre-wrap wrap-break-word", logColor(e.level))}
                >
                  <span className="mr-2 text-muted-foreground">
                    {new Date(e.ts).toLocaleTimeString()}
                  </span>
                  {e.message}
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* Live results table (only while running) */}
      {isEvalRunning && liveResults.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Live results</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="max-h-80 overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-16">Idx</TableHead>
                    <TableHead>Score</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {liveResults
                    .slice()
                    .sort((a, b) => a.idx - b.idx)
                    .map((r) => (
                      <TableRow key={r.idx}>
                        <TableCell className="tabular-nums">{r.idx}</TableCell>
                        <TableCell>
                          {r.error ? (
                            <Badge variant="destructive">error</Badge>
                          ) : (
                            <span className={cn("tabular-nums font-medium", scoreColor(r.score))}>
                              {r.score.toFixed(3)}
                            </span>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Full run detail once terminal */}
      {isTerminal && <EvalRunDetail runId={currentRunId} />}
    </div>
  );
}
