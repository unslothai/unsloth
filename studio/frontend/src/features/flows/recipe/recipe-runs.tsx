// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A recipe group's runs in Recipe Studio's own executions view: rows, columns, overview, download and publish.
// Lazy: none of it loads until Preview or Runs is opened.

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  cancelRecipeJob,
  getRecipeJobDataset,
} from "@/features/recipe-studio/api";
import { ExecutionsView } from "@/features/recipe-studio/components/executions/executions-view";
import type { RecipeExecutionRecord } from "@/features/recipe-studio/execution-types";
import {
  normalizeDatasetRows,
  toErrorMessage,
} from "@/features/recipe-studio/executions/execution-helpers";
import { toastError } from "@/shared/toast";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { recipeRuns } from "../api";
import { runFlow } from "../editor/run-controller";
import { useDocStore } from "../stores/doc-store";
import { useRuntimeStore } from "../stores/runtime-store";
import { type RecipeRunsTarget, useUiStore } from "../stores/ui-store";
import { groupGenerate } from "./recipe-columns";
import {
  runsAreLive,
  selectedRun,
  toRunRecords,
  visibleRuns,
} from "./recipe-runs-state";

const POLL_MS = 2000;
const RUNS_LIMIT = 200;

type Page = Pick<
  RecipeExecutionRecord,
  "dataset" | "datasetPage" | "datasetTotal"
>;

function useRecipeRuns(flowId: string | null, nodeId: string | undefined) {
  const [records, setRecords] = useState<RecipeExecutionRecord[] | null>(null);
  const [pages, setPages] = useState<Record<string, Page>>({});
  const flowRunning = useRuntimeStore(
    (s) => s.runStatus === "running" || s.runStatus === "queued",
  );
  const warned = useRef(false);
  const load = useCallback(async () => {
    if (!flowId || !nodeId) return;
    try {
      const { executions } = await recipeRuns(flowId, nodeId, RUNS_LIMIT);
      setRecords(toRunRecords(executions));
      warned.current = false;
    } catch (error) {
      setRecords((current) => current ?? []);
      if (!warned.current)
        toastError("Could not load the runs", toErrorMessage(error, ""));
      warned.current = true;
    }
  }, [flowId, nodeId]);
  const live = runsAreLive(records ?? [], flowRunning);
  useEffect(() => {
    void load();
    if (!live) return;
    const timer = setInterval(() => void load(), POLL_MS);
    return () => clearInterval(timer);
  }, [load, live]);
  const merged = useMemo(
    () =>
      (records ?? []).map((record) =>
        pages[record.id] ? { ...record, ...pages[record.id] } : record,
      ),
    [records, pages],
  );
  const loadPage = useCallback(
    async (id: string, page: number) => {
      const record = merged.find((r) => r.id === id);
      if (!record?.jobId || record.kind !== "full" || page < 1) return;
      const size = record.datasetPageSize || 20;
      try {
        const response = await getRecipeJobDataset(record.jobId, {
          limit: size,
          offset: (page - 1) * size,
        });
        setPages((current) => ({
          ...current,
          [id]: {
            dataset: normalizeDatasetRows(response.dataset),
            datasetPage: page,
            datasetTotal:
              typeof response.total === "number"
                ? response.total
                : record.datasetTotal,
          },
        }));
      } catch (error) {
        toastError(
          "Dataset page failed",
          toErrorMessage(error, "Could not load dataset page."),
        );
      }
    },
    [merged],
  );
  const cancel = useCallback(
    async (id: string) => {
      const record = merged.find((r) => r.id === id);
      if (!record?.jobId) return;
      try {
        await cancelRecipeJob(record.jobId);
        await load();
      } catch (error) {
        toastError(
          "Cancel failed",
          toErrorMessage(error, "Could not cancel execution."),
        );
      }
    },
    [merged, load],
  );
  return { records: records === null ? null : merged, loadPage, cancel };
}

function RecipeRunsOpen({ target }: { target: RecipeRunsTarget }) {
  const flowId = useDocStore((s) => s.flowId);
  const group = useDocStore((s) => s.graph.groups[target.groupId]);
  const step = useDocStore((s) => groupGenerate(s.graph, target.groupId));
  const busy = useRuntimeStore(
    (s) =>
      s.runStatus === "running" ||
      s.runStatus === "queued" ||
      s.runStatus === "cancelling",
  );
  const { records, loadPage, cancel } = useRecipeRuns(flowId, step?.id);
  const [selected, setSelected] = useState<string | null>(null);
  const preview = target.view === "preview";
  const shown = useMemo(
    () => visibleRuns(records ?? [], target.view),
    [records, target.view],
  );
  const selectedId = selectedRun(shown, selected);
  const close = () => useUiStore.getState().setRecipeRuns(null);
  const runAgain = () => {
    if (!step) return;
    void runFlow({ targetNodes: [step.id], forceNodes: [step.id] });
  };
  const title = `${group?.label ?? "Data Recipe"}: ${preview ? "Preview" : "Runs"}`;
  return (
    <Dialog open={true} onOpenChange={(open) => !open && close()}>
      <DialogContent
        className="flex h-[min(85vh,860px)] flex-col gap-0 overflow-hidden p-0 sm:max-w-[min(92vw,1280px)]"
        data-testid="flows-recipe-runs"
      >
        <div className="flex items-center gap-3 border-b px-4 py-3 pr-12">
          <div className="min-w-0 flex-1">
            <DialogTitle className="truncate text-base">{title}</DialogTitle>
            <DialogDescription className="text-xs">
              {preview
                ? "The latest run of this recipe: its rows, each column, and any errors."
                : `This recipe's runs in this flow, newest first (up to ${RUNS_LIMIT}).`}
            </DialogDescription>
          </div>
          {preview && (records?.length ?? 0) > 1 && (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={() =>
                useUiStore
                  .getState()
                  .setRecipeRuns({ groupId: target.groupId, view: "runs" })
              }
            >
              All runs ({records?.length})
            </Button>
          )}
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={!step || busy}
            onClick={runAgain}
          >
            {busy ? "Running…" : "Run again"}
          </Button>
        </div>
        <div className="min-h-0 flex-1">
          {records === null ? (
            <p className="p-6 text-sm text-muted-foreground">Loading runs…</p>
          ) : shown.length === 0 ? (
            <p className="p-6 text-sm text-muted-foreground">
              This recipe has not run yet. Run the flow, or press Run again, and
              its rows show up here.
            </p>
          ) : (
            <ExecutionsView
              executions={shown}
              selectedExecutionId={selectedId}
              currentSignature=""
              onSelectExecution={setSelected}
              onCancelExecution={(id) => void cancel(id)}
              onLoadDatasetPage={(id, page) => void loadPage(id, page)}
            />
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default function RecipeRuns() {
  const target = useUiStore((s) => s.recipeRuns);
  if (!target) return null;
  return <RecipeRunsOpen key={JSON.stringify(target)} target={target} />;
}
