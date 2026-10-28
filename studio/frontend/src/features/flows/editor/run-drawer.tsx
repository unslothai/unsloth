// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import { useReactFlow } from "@xyflow/react";
import { memo, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { GENERATE_TYPE } from "../recipe/recipe-columns";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import {
  type NodeRuntime,
  type RunLogLine,
  useRuntimeStore,
} from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import { isDatasetValue } from "../dataset/dataset-value";
import type { DatasetValue, MediaValue } from "../types";
import { DatasetOutput } from "./dataset-output";
import { ArrowDown01Icon } from "./icon-data";
import { MediaThumb, isMediaValue } from "./media";
import { StructuredOutput, isStructuredValue } from "./value-viewers";

const RUN_STATUS_STYLE: Record<string, string> = {
  queued: "bg-zinc-700 text-zinc-200",
  running: "bg-sky-600 text-white",
  cancelling: "bg-amber-600 text-white",
  cancelled: "bg-zinc-600 text-zinc-100",
  completed: "bg-emerald-600 text-white",
  failed: "bg-red-600 text-white",
  interrupted: "bg-amber-700 text-white",
};

function duration(rt: NodeRuntime, now: number): string {
  if (!rt.startedAt) return "";
  const end = rt.finishedAt ?? now / 1000;
  const s = Math.max(0, end - rt.startedAt);
  return s < 60
    ? `${s.toFixed(1)}s`
    : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function Timeline() {
  const runtimeNodes = useRuntimeStore((s) => s.nodes);
  const graph = useDocStore((s) => s.graph);
  const specs = useCatalogStore((s) => s.specs);
  const { fitView } = useReactFlow();
  const rows = useMemo(
    () =>
      graph.nodeOrder
        .map((id) => ({ id, node: graph.nodes[id], rt: runtimeNodes[id] }))
        .filter((row) => row.rt && row.rt.status !== "idle")
        .sort(
          (a, b) =>
            (a.rt.startedAt ?? Number.POSITIVE_INFINITY) -
            (b.rt.startedAt ?? Number.POSITIVE_INFINITY),
        ),
    [graph, runtimeNodes],
  );
  if (!rows.length)
    return (
      <p className="px-4 py-6 text-center text-[12px] text-zinc-500">
        Run the flow to see each step here.
      </p>
    );
  const now = Date.now();
  return (
    <ul className="divide-y divide-white/5">
      {rows.map(({ id, node, rt }) => (
        <TimelineRow
          key={id}
          id={id}
          title={node.label || specs[node.type]?.title || node.type}
          rt={rt}
          now={rt.startedAt && rt.finishedAt == null ? now : 0}
          fitView={fitView}
          recipeGroup={
            node.type === GENERATE_TYPE ? (node.group ?? undefined) : undefined
          }
        />
      ))}
    </ul>
  );
}

const TimelineRow = memo(function TimelineRow({
  id,
  title,
  rt,
  now,
  fitView,
  recipeGroup,
}: {
  id: string;
  title: string;
  rt: NodeRuntime;
  now: number;
  fitView: ReturnType<typeof useReactFlow>["fitView"];
  recipeGroup?: string;
}) {
  return (
    <li className="flex items-center hover:bg-white/5">
      <button
        type="button"
        onClick={() => {
          useUiStore.getState().setSelection([id], []);
          useUiStore.getState().openInspector(id);
          void fitView({ nodes: [{ id }], duration: 300, maxZoom: 1.2 });
        }}
        className="flex min-w-0 flex-1 items-center gap-3 px-4 py-1.5 text-left text-[12.5px]"
      >
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${
            rt.status === "failed"
              ? "bg-red-500"
              : rt.status === "running" || rt.status === "loading_model"
                ? "animate-pulse bg-sky-400"
                : rt.status === "done" || rt.status === "cached"
                  ? "bg-emerald-400"
                  : "bg-zinc-500"
          }`}
        />
        <span className="w-48 truncate text-zinc-200">{title}</span>
        <span className="w-24 text-zinc-500">
          {rt.cached ? "cached" : rt.status.replace("_", " ")}
        </span>
        {rt.status === "failed" ? (
          <span className="min-w-0 flex-1 text-red-300">
            {rt.error || "It stopped with an error"}{" "}
            <span className="whitespace-nowrap text-zinc-400 underline underline-offset-2">
              Open block
            </span>
          </span>
        ) : (
          <span className="min-w-0 flex-1 truncate text-zinc-500">
            {rt.error ?? rt.phase ?? ""}
          </span>
        )}
        {rt.progress !== null && rt.status === "running" && (
          <span className="h-1 w-24 overflow-hidden rounded bg-white/10">
            <span
              className="block h-full bg-sky-400"
              style={{ width: `${Math.round(rt.progress * 100)}%` }}
            />
          </span>
        )}
        <span className="w-14 text-right font-mono text-[11px] text-zinc-500">
          {duration(rt, now)}
        </span>
      </button>
      {recipeGroup &&
        (["preview", "runs"] as const).map((view) => (
          <button
            key={view}
            type="button"
            onClick={() =>
              useUiStore
                .getState()
                .setRecipeRuns({ groupId: recipeGroup, view })
            }
            className="mr-2 rounded px-1.5 py-0.5 text-[11.5px] capitalize text-sky-300 hover:bg-white/10"
          >
            {view}
          </button>
        ))}
    </li>
  );
});

function Logs() {
  const logs = useRuntimeStore((s) => s.logs);
  if (!logs.length)
    return (
      <p className="px-4 py-6 text-center text-[12px] text-zinc-500">
        No log lines yet.
      </p>
    );
  return (
    <pre className="px-4 py-2 font-mono text-[11px] leading-relaxed text-zinc-400">
      {logs.map((line) => (
        <LogLine key={line.seq} line={line} />
      ))}
    </pre>
  );
}

const LogLine = memo(function LogLine({ line }: { line: RunLogLine }) {
  return (
    <div
      className={
        line.level === "error"
          ? "text-red-300"
          : line.level === "warning"
            ? "text-amber-300"
            : ""
      }
    >
      <span className="text-zinc-600">
        {new Date(line.ts * 1000).toLocaleTimeString()}{" "}
      </span>
      {line.nodeId ? (
        <span className="text-zinc-500">[{line.nodeId}] </span>
      ) : null}
      {line.message}
    </div>
  );
});

function Outputs() {
  const flat = useRuntimeStore(
    useShallow((s) =>
      Object.entries(s.nodes).flatMap(([id, rt]) => [id, rt.outputs]),
    ),
  );
  const runtimeOutputs = useMemo(() => {
    const pairs: [string, Record<string, unknown> | null][] = [];
    for (let i = 0; i < flat.length; i += 2)
      pairs.push([
        flat[i] as string,
        flat[i + 1] as Record<string, unknown> | null,
      ]);
    return pairs;
  }, [flat]);
  const media = useMemo(() => {
    const items: { key: string; media: MediaValue }[] = [];
    for (const [id, outputs] of runtimeOutputs) {
      for (const [port, value] of Object.entries(outputs ?? {})) {
        const list = Array.isArray(value) ? value : [value];
        list.forEach((item, i) => {
          if (isMediaValue(item))
            items.push({ key: `${id}:${port}:${i}:${item.ref}`, media: item });
        });
      }
    }
    const seen = new Set<string>();
    return items
      .filter((item) => {
        if (seen.has(item.media.ref)) return false;
        seen.add(item.media.ref);
        return true;
      })
      .slice(0, 120);
  }, [runtimeOutputs]);
  const structured = useMemo(() => {
    const items: { key: string; label: string; value: unknown }[] = [];
    for (const [id, outputs] of runtimeOutputs) {
      for (const [port, value] of Object.entries(outputs ?? {})) {
        if (isStructuredValue(value))
          items.push({ key: `${id}:${port}`, label: `${id} / ${port}`, value });
      }
    }
    return items.slice(0, 12);
  }, [runtimeOutputs]);
  const datasets = useMemo(() => {
    const items: { key: string; label: string; value: DatasetValue }[] = [];
    const seen = new Set<string>();
    for (const [id, outputs] of runtimeOutputs) {
      for (const [port, value] of Object.entries(outputs ?? {})) {
        if (!isDatasetValue(value) || seen.has(value.ref)) continue;
        seen.add(value.ref);
        items.push({ key: `${id}:${port}`, label: `${id} / ${port}`, value });
      }
    }
    return items.slice(0, 12);
  }, [runtimeOutputs]);
  if (!media.length && !structured.length && !datasets.length)
    return (
      <p className="px-4 py-6 text-center text-[12px] text-zinc-500">
        Images, audio, video, tables, charts and datasets from the run appear
        here.
      </p>
    );
  return (
    <div className="flex flex-col gap-3 p-3">
      {datasets.map((item) => (
        <section key={item.key} className="flex max-w-xl flex-col gap-1">
          <h4 className="font-mono text-[11px] text-zinc-500">{item.label}</h4>
          <DatasetOutput value={item.value} compact />
        </section>
      ))}
      {structured.map((item) => (
        <section key={item.key} className="flex flex-col gap-1">
          <h4 className="font-mono text-[11px] text-zinc-500">{item.label}</h4>
          <StructuredOutput value={item.value} />
        </section>
      ))}
      {media.length ? (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-2">
          {media.map((item) => (
            <MediaThumb
              key={item.key}
              media={item.media}
              className="aspect-square w-full rounded-md object-cover"
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function RunDrawerImpl() {
  const open = useUiStore((s) => s.drawerOpen);
  const tab = useUiStore((s) => s.drawerTab);
  const runStatus = useRuntimeStore((s) => s.runStatus);
  const runError = useRuntimeStore((s) => s.runError);
  const setDrawer = useUiStore((s) => s.setDrawer);
  if (!open) return null;
  return (
    <div
      className="flex h-65 shrink-0 flex-col border-white/10 border-t bg-zinc-950/95 text-zinc-100"
      data-testid="flows-run-drawer"
    >
      <div className="flex items-center gap-1 border-white/10 border-b px-2 py-1">
        {(["run", "outputs", "logs"] as const).map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setDrawer(true, id)}
            className={`rounded-md px-2.5 py-1 text-[12px] capitalize ${tab === id ? "bg-white/10 text-white" : "text-zinc-400 hover:text-zinc-200"}`}
          >
            {id}
          </button>
        ))}
        {runStatus && (
          <span
            className={`ml-2 rounded px-1.5 py-0.5 text-[10.5px] capitalize ${RUN_STATUS_STYLE[runStatus] ?? ""}`}
          >
            {runStatus}
          </span>
        )}
        {runError && (
          <span className="ml-2 truncate text-[11.5px] text-red-300">
            {runError}
          </span>
        )}
        <button
          type="button"
          aria-label="Hide run panel"
          onClick={() => setDrawer(false)}
          className="ml-auto rounded p-1 text-zinc-500 hover:bg-white/5 hover:text-zinc-200"
        >
          <HugeiconsIcon icon={ArrowDown01Icon} size={14} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {tab === "run" ? <Timeline /> : tab === "logs" ? <Logs /> : <Outputs />}
      </div>
    </div>
  );
}

export const RunDrawer = memo(RunDrawerImpl);
