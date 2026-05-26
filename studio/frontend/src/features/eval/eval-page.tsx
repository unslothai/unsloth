// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { toast } from "@/lib/toast";
import { startEval, listEvalRuns, type EvalStartRequest, type EvalRunSummary } from "./api/eval-api";
import { EvalConfigForm } from "./components/eval-config-form";
import { EvalRunDetail } from "./components/eval-run-detail";
import { LiveEvalView } from "./components/live-eval-view";
import { useEvalProgressStream } from "./hooks/use-eval-progress-stream";
import {
  emitEvalRunsChanged,
  EVAL_RUNS_CHANGED,
} from "./hooks/use-eval-history-sidebar";
import { useEvalRuntimeStore } from "./stores/eval-runtime-store";
import type { EvalStatus } from "./api/eval-api";

// ── Badge variant helper ─────────────────────────────────────────────────

function statusBadgeVariant(
  status: EvalStatus,
): "default" | "secondary" | "destructive" | "outline" {
  if (status === "completed") return "default";
  if (status === "running") return "secondary";
  if (status === "error") return "destructive";
  return "outline";
}

// ── EvalHistoryList ───────────────────────────────────────────────────────

function EvalHistoryList({ onSelect }: { onSelect: (id: string) => void }) {
  const [items, setItems] = useState<EvalRunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    const load = () =>
      listEvalRuns()
        .then((res) => {
          if (mounted) setItems(res.runs);
        })
        .catch(() => {})
        .finally(() => {
          if (mounted) setLoading(false);
        });
    load();
    const onChange = () => load();
    window.addEventListener(EVAL_RUNS_CHANGED, onChange);
    return () => {
      mounted = false;
      window.removeEventListener(EVAL_RUNS_CHANGED, onChange);
    };
  }, []);

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading…</p>;
  }

  if (items.length === 0) {
    return <p className="text-sm text-muted-foreground">No eval runs yet.</p>;
  }

  return (
    <div className="flex flex-col gap-3">
      {items.map((run) => (
        <Card
          key={run.id}
          className="cursor-pointer transition-colors hover:bg-muted/50"
          onClick={() => onSelect(run.id)}
        >
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between gap-2 text-base">
              <span className="truncate">
                {run.display_name ?? run.model_identifier}
              </span>
              <Badge variant={statusBadgeVariant(run.status)}>
                {run.status}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-between gap-4 text-sm text-muted-foreground">
            <span className="truncate">
              {run.metric_name} · {run.dataset_ref}
            </span>
            <div className="flex shrink-0 flex-col items-end gap-0.5">
              <span className="tabular-nums font-medium text-foreground">
                {run.avg_score != null ? run.avg_score.toFixed(4) : "—"}
              </span>
              <span className="text-xs">
                {new Date(run.started_at).toLocaleString()}
              </span>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ── EvalPage ──────────────────────────────────────────────────────────────

export function EvalPage() {
  const isEvalRunning = useEvalRuntimeStore((s) => s.isEvalRunning);
  const currentRunId = useEvalRuntimeStore((s) => s.currentRunId);
  const selectedHistoryRunId = useEvalRuntimeStore(
    (s) => s.selectedHistoryRunId,
  );
  const startError = useEvalRuntimeStore((s) => s.startError);
  const beginRun = useEvalRuntimeStore((s) => s.beginRun);
  const setStartError = useEvalRuntimeStore((s) => s.setStartError);
  const setSelectedHistoryRunId = useEvalRuntimeStore(
    (s) => s.setSelectedHistoryRunId,
  );

  const [tab, setTab] = useState<"configure" | "run" | "history">("configure");

  // Keep the SSE stream alive while on the page
  useEvalProgressStream(currentRunId, isEvalRunning);

  // Sidebar-selection → History tab effect
  useEffect(() => {
    if (selectedHistoryRunId && selectedHistoryRunId !== currentRunId) {
      setTab("history");
    }
  }, [selectedHistoryRunId, currentRunId]);

  // Refresh sidebar when a run finishes
  const prevRunning = useRef(false);
  useEffect(() => {
    if (prevRunning.current && !isEvalRunning) emitEvalRunsChanged();
    prevRunning.current = isEvalRunning;
  }, [isEvalRunning]);

  async function handleStart(payload: EvalStartRequest) {
    setStartError(null);
    try {
      const { run_id } = await startEval(payload);
      beginRun(run_id, payload.limit ?? 0);
      emitEvalRunsChanged();
      setTab("run");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStartError(msg);
      toast.error("Could not start eval", { description: msg });
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 p-6 overflow-y-auto">
      <div className="flex flex-col gap-1">
        <h1 className="font-heading text-2xl font-semibold tracking-tight">Eval</h1>
        <p className="text-sm text-muted-foreground">
          Evaluate a model (trained checkpoint, Hugging Face repo, or local) against a
          dataset with a task-appropriate metric, then inspect per-example results.
        </p>
      </div>

      <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)}>
        <TabsList>
          <TabsTrigger value="configure">Configure</TabsTrigger>
          <TabsTrigger value="run">Run</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        <TabsContent value="configure" className="mt-4">
          <EvalConfigForm disabled={isEvalRunning} onStart={handleStart} />
          {startError ? (
            <p className="mt-3 text-sm text-red-500">{startError}</p>
          ) : null}
        </TabsContent>

        <TabsContent value="run" className="mt-4">
          <LiveEvalView />
        </TabsContent>

        <TabsContent value="history" className="mt-4">
          {selectedHistoryRunId ? (
            <div className="flex flex-col gap-3">
              <Button
                variant="ghost"
                size="sm"
                className="self-start"
                onClick={() => setSelectedHistoryRunId(null)}
              >
                ← Back to all runs
              </Button>
              <EvalRunDetail runId={selectedHistoryRunId} />
            </div>
          ) : (
            <EvalHistoryList onSelect={(id) => setSelectedHistoryRunId(id)} />
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
