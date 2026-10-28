// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription } from "@/components/ui/empty";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HugeiconsIcon } from "@hugeicons/react";
import { useReactFlow } from "@xyflow/react";
import { memo, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import {
  type NodeRuntime,
  type RunLogLine,
  useRuntimeStore,
} from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import type { MediaValue } from "../types";
import { ArrowDown01Icon } from "./icon-data";
import { MediaThumb, isMediaValue } from "./media";
import { StructuredOutput, isStructuredValue } from "./value-viewers";

const RUN_STATUS_STYLE: Record<string, string> = {
  queued: "text-muted-foreground",
  running: "border-border bg-muted text-foreground",
  cancelling:
    "border-amber-500/25 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  cancelled: "text-muted-foreground",
  completed:
    "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  failed: "border-destructive/25 bg-destructive/10 text-destructive",
  interrupted:
    "border-amber-500/25 bg-amber-500/10 text-amber-700 dark:text-amber-300",
};

function DrawerEmpty({ children }: { children: string }) {
  return (
    <Empty className="p-6">
      <EmptyDescription className="text-xs">{children}</EmptyDescription>
    </Empty>
  );
}

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
    return <DrawerEmpty>Run the flow to see each step here.</DrawerEmpty>;
  const now = Date.now();
  return (
    <ul className="divide-y divide-border/60">
      {rows.map(({ id, node, rt }) => (
        <TimelineRow
          key={id}
          id={id}
          title={node.label || specs[node.type]?.title || node.type}
          rt={rt}
          now={rt.startedAt && rt.finishedAt == null ? now : 0}
          fitView={fitView}
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
}: {
  id: string;
  title: string;
  rt: NodeRuntime;
  now: number;
  fitView: ReturnType<typeof useReactFlow>["fitView"];
}) {
  return (
    <li>
      <button
        type="button"
        onClick={() => {
          useUiStore.getState().setSelection([id], []);
          useUiStore.getState().openInspector(id);
          void fitView({ nodes: [{ id }], duration: 300, maxZoom: 1.2 });
        }}
        className="flex w-full items-center gap-3 px-4 py-1.5 text-left text-xs transition hover:bg-muted/35"
      >
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${
            rt.status === "failed"
              ? "bg-destructive"
              : rt.status === "running" || rt.status === "loading_model"
                ? "animate-pulse bg-primary"
                : rt.status === "done" || rt.status === "cached"
                  ? "bg-emerald-500"
                  : "bg-muted-foreground"
          }`}
        />
        <span className="w-48 truncate text-foreground">{title}</span>
        <span className="w-24 text-muted-foreground">
          {rt.cached ? "cached" : rt.status.replace("_", " ")}
        </span>
        {rt.status === "failed" ? (
          <span className="min-w-0 flex-1 text-destructive">
            {rt.error || "It stopped with an error"}{" "}
            <span className="whitespace-nowrap text-muted-foreground underline underline-offset-2">
              Open block
            </span>
          </span>
        ) : (
          <span className="min-w-0 flex-1 truncate text-muted-foreground">
            {rt.error ?? rt.phase ?? ""}
          </span>
        )}
        {rt.progress !== null && rt.status === "running" && (
          <Progress
            value={Math.round(rt.progress * 100)}
            className="h-1 w-24"
          />
        )}
        <span className="w-14 text-right font-mono text-ui-11 text-muted-foreground">
          {duration(rt, now)}
        </span>
      </button>
    </li>
  );
});

function Logs() {
  const logs = useRuntimeStore((s) => s.logs);
  if (!logs.length)
    return <DrawerEmpty>No log lines yet.</DrawerEmpty>;
  return (
    <pre className="px-4 py-2 font-mono text-ui-11 leading-relaxed text-muted-foreground">
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
          ? "text-destructive"
          : line.level === "warning"
            ? "text-amber-700 dark:text-amber-300"
            : ""
      }
    >
      <span className="text-muted-foreground/70">
        {new Date(line.ts * 1000).toLocaleTimeString()}{" "}
      </span>
      {line.nodeId ? (
        <span className="text-muted-foreground">[{line.nodeId}] </span>
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
  if (!media.length && !structured.length)
    return (
      <DrawerEmpty>
        Images, audio, video, tables and charts from the run appear here.
      </DrawerEmpty>
    );
  return (
    <div className="flex flex-col gap-3 p-3">
      {structured.map((item) => (
        <section key={item.key} className="flex flex-col gap-1">
          <h4 className="font-mono text-ui-11 text-muted-foreground">{item.label}</h4>
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
      className="flex h-65 shrink-0 flex-col border-border border-t bg-background text-foreground"
      data-testid="flows-run-drawer"
    >
      <div className="flex items-center gap-2 border-border/60 border-b px-2 py-1.5">
        <Tabs
          value={tab}
          onValueChange={(value) => setDrawer(true, value as typeof tab)}
        >
          <TabsList className="group-data-horizontal/tabs:h-8">
            {(["run", "outputs", "logs"] as const).map((id) => (
              <TabsTrigger key={id} value={id} className="px-3 text-xs capitalize">
                {id}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        {runStatus && (
          <Badge
            variant="outline"
            className={`capitalize ${RUN_STATUS_STYLE[runStatus] ?? ""}`}
          >
            {runStatus}
          </Badge>
        )}
        {runError && (
          <span className="truncate text-xs text-destructive">
            {runError}
          </span>
        )}
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label="Hide run panel"
          onClick={() => setDrawer(false)}
          className="ml-auto text-muted-foreground"
        >
          <HugeiconsIcon icon={ArrowDown01Icon} size={14} />
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {tab === "run" ? <Timeline /> : tab === "logs" ? <Logs /> : <Outputs />}
      </div>
    </div>
  );
}

export const RunDrawer = memo(RunDrawerImpl);
