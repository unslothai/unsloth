// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import { Handle, type NodeProps, Position, useStore } from "@xyflow/react";
import { memo } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import {
  IDLE_RUNTIME,
  type NodeRuntime,
  useRuntimeStore,
} from "../stores/runtime-store";
import { COLUMN_TYPE, columnOf, columnSummary } from "../recipe/recipe-columns";
import { useColumnSamples } from "../recipe/recipe-panels";
import type { FlowNode, NodeSpec, PortSpec } from "../types";
import {
  FLOW_IN_HANDLE,
  flowOutHandle,
  inHandle,
  outHandle,
} from "./connection-rules";
import type { RfNode } from "./derive-rf";
import { nodeIcon } from "./icons";
import { MediaThumb, isMediaValue } from "./media";
import { TrainNodeBody } from "./train-node";
import { StructuredBadge, isStructuredValue } from "./value-viewers";

/** Below this zoom the text is unreadable anyway; hiding it (layout kept, so handles stay put)
 *  is what keeps panning a thousand-node graph smooth. */
const LOW_DETAIL_ZOOM = 0.45;
const lowDetailSelector = (s: { transform: [number, number, number] }) =>
  s.transform[2] < LOW_DETAIL_ZOOM;

const STATUS_RING: Record<string, string> = {
  queued: "ring-1 ring-sky-400/50",
  waiting: "ring-2 ring-amber-400/70",
  loading_model: "ring-2 ring-sky-400 animate-pulse",
  running: "ring-2 ring-sky-400 shadow-[0_0_24px_-4px] shadow-sky-500/60",
  done: "ring-1 ring-emerald-400/70",
  cached: "ring-1 ring-emerald-400/40",
  failed: "ring-2 ring-red-500",
  cancelled: "ring-1 ring-zinc-500",
  skipped: "opacity-60",
};

const STATUS_LABEL: Record<string, string> = {
  queued: "Queued",
  waiting: "Waiting",
  loading_model: "Loading model",
  running: "Running",
  done: "Done",
  cached: "Cached",
  failed: "Failed",
  cancelled: "Cancelled",
  skipped: "Skipped",
};

function portColorOf(
  colors: Record<string, string> | undefined,
  port: PortSpec,
): string {
  return colors?.[port.type] ?? "#cbd5e1";
}

function summarizeParams(node: FlowNode, spec: NodeSpec): string[] {
  if (node.type === COLUMN_TYPE) {
    const { badge, line } = columnSummary(columnOf(node));
    return line ? [badge, line] : [badge];
  }
  const lines: string[] = [];
  for (const param of spec.params) {
    if (lines.length >= 2) break;
    const value = node.params[param.id];
    if (value === null || value === undefined || value === "") continue;
    if (param.kind === "model") {
      const repo = (value as { repo?: string }).repo;
      if (repo) lines.push(repo.split("/").pop() ?? repo);
      continue;
    }
    if (param.kind === "textarea" || param.kind === "text") {
      const text = String(value).trim();
      if (text) lines.push(text.length > 90 ? `${text.slice(0, 90)}…` : text);
      continue;
    }
    if (param.kind === "select" && lines.length < 2)
      lines.push(`${param.label}: ${String(value)}`);
  }
  return lines;
}

function OutputPreview({
  runtime,
  spec,
}: { runtime: NodeRuntime; spec: NodeSpec }) {
  if (
    runtime.preview &&
    (runtime.status === "running" || runtime.status === "loading_model")
  ) {
    return (
      <img
        src={runtime.preview}
        alt=""
        className="aspect-square w-full rounded-md object-cover"
        draggable={false}
      />
    );
  }
  if ((runtime.status === "running" || !runtime.outputs) && runtime.text) {
    return (
      <p className="line-clamp-4 whitespace-pre-wrap text-[11px] leading-snug text-zinc-300">
        {runtime.text}
      </p>
    );
  }
  if (
    runtime.partial &&
    runtime.status !== "done" &&
    runtime.status !== "cached"
  ) {
    const at = runtime.partialStep;
    return (
      <div className="flex flex-col gap-1">
        <span className="text-[10px] uppercase tracking-wide text-amber-300/80">
          {at
            ? runtime.status === "running"
              ? `Partial: ${at.step} of ${at.total}`
              : `Partial (stopped at ${at.step} of ${at.total})`
            : "Partial"}
        </span>
        <OutputValues outputs={runtime.partial} spec={spec} />
      </div>
    );
  }
  const outputs = runtime.outputs;
  if (!outputs) return null;
  return <OutputValues outputs={outputs} spec={spec} />;
}

function OutputValues({
  outputs,
  spec,
}: { outputs: Record<string, unknown>; spec: NodeSpec }) {
  for (const port of spec.outputs) {
    const value = outputs[port.id];
    if (isMediaValue(value)) {
      return (
        <MediaThumb
          media={value}
          className="max-h-48 w-full rounded-md object-cover"
        />
      );
    }
    if (Array.isArray(value) && value.length && isMediaValue(value[0])) {
      return (
        <div className="grid grid-cols-3 gap-1">
          {value
            .slice(0, 6)
            .map((item, i) =>
              isMediaValue(item) ? (
                <MediaThumb
                  key={`${item.ref}-${i}`}
                  media={item}
                  className="aspect-square w-full rounded object-cover"
                />
              ) : null,
            )}
        </div>
      );
    }
  }
  for (const port of spec.outputs) {
    const value = outputs[port.id];
    if (value === undefined || value === null) continue;
    if (isStructuredValue(value)) return <StructuredBadge value={value} />;
    if (typeof value === "string") {
      return (
        <p className="line-clamp-4 whitespace-pre-wrap text-[11px] leading-snug text-zinc-300">
          {value}
        </p>
      );
    }
    if (typeof value === "number" || typeof value === "boolean") {
      return (
        <span className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-[11px] text-zinc-200">
          {String(value)}
        </span>
      );
    }
    if (Array.isArray(value)) {
      return (
        <span className="rounded bg-white/5 px-1.5 py-0.5 text-[11px] text-zinc-300">
          {value.length} items
        </span>
      );
    }
    return (
      <span className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-[11px] text-zinc-300">
        {"{ "}
        {Object.keys(value as object)
          .slice(0, 4)
          .join(", ")}
        {" }"}
      </span>
    );
  }
  return null;
}

/** The value this column got in the first row of its recipe's last run. */
function ColumnSample({ node }: { node: FlowNode }) {
  const [sample] = useColumnSamples(node, 1);
  if (!sample) return null;
  return (
    <p className="flow-detail mx-3 line-clamp-3 whitespace-pre-wrap rounded bg-emerald-500/10 px-2 py-1 text-[11px] leading-snug text-emerald-100/90">
      {sample}
    </p>
  );
}

function FlowNodeViewImpl({ id, selected }: NodeProps<RfNode>) {
  const node = useDocStore((s) => s.graph.nodes[id]);
  const spec = useCatalogStore((s) => (node ? s.specs[node.type] : undefined));
  const color =
    useCatalogStore((s) =>
      spec ? s.categories[spec.category]?.color : undefined,
    ) ?? "#64748b";
  const portColors = useCatalogStore((s) => s.catalog?.port_colors);
  const runtime = useRuntimeStore((s) => s.nodes[id]) ?? IDLE_RUNTIME;
  const lowDetail = useStore(lowDetailSelector);
  if (!node) return null;
  if (!spec) {
    return (
      <div className="w-60 rounded-xl border border-red-500/60 bg-zinc-900 p-3 text-xs text-red-300">
        Unknown node type <code>{node.type}</code>
      </div>
    );
  }
  const title = node.label || spec.title;
  const ring = STATUS_RING[runtime.status] ?? "";
  const running =
    runtime.status === "running" || runtime.status === "loading_model";
  const summary = node.collapsed ? [] : summarizeParams(node, spec);
  const collapsed = !!node.collapsed;

  return (
    <div
      className={`group/node relative w-65 rounded-xl border bg-zinc-900 text-zinc-100 shadow-md shadow-black/30 transition-opacity ${
        selected ? "border-white/60" : "border-white/10"
      } ${ring} ${node.disabled ? "opacity-45" : ""} ${lowDetail ? "[&_.flow-detail]:invisible" : ""}`}
      data-node-id={id}
      data-status={runtime.status}
    >
      <div
        className="flow-node-drag flex cursor-grab items-center gap-2 rounded-t-xl px-3 py-2 active:cursor-grabbing"
        style={{
          background: `linear-gradient(90deg, ${color}33, transparent 80%)`,
          borderTop: `2px solid ${color}`,
        }}
      >
        {spec.flow_in && (
          <Handle
            type="target"
            id={FLOW_IN_HANDLE}
            position={Position.Left}
            className="!h-3 !w-3 !rounded-sm !border-2 !border-white !bg-zinc-900"
            style={{ top: 18 }}
          />
        )}
        <HugeiconsIcon
          icon={nodeIcon(spec.icon)}
          size={15}
          strokeWidth={1.8}
          style={{ color }}
        />
        <span className="min-w-0 flex-1 truncate font-medium text-[13px]">
          {title}
        </span>
        {runtime.cached && runtime.status !== "running" && (
          <span className="rounded bg-emerald-500/15 px-1 text-[10px] text-emerald-300">
            cached
          </span>
        )}
        {runtime.status !== "idle" && !runtime.cached && (
          <span
            className={`text-[10px] ${runtime.status === "failed" ? "text-red-300" : running ? "text-sky-300" : "text-zinc-400"}`}
            title={runtime.error ?? undefined}
          >
            {STATUS_LABEL[runtime.status] ?? runtime.status}
          </span>
        )}
        {spec.lane === "gpu" && runtime.status === "idle" && (
          <span className="rounded bg-white/5 px-1 text-[9px] uppercase tracking-wide text-zinc-500">
            gpu
          </span>
        )}
      </div>

      {collapsed ? (
        <div className="relative h-0">
          {spec.inputs.map((port) => (
            <Handle
              key={port.id}
              type="target"
              id={inHandle(port.id)}
              position={Position.Left}
              className="!opacity-0"
              style={{ top: -18 }}
            />
          ))}
          {spec.outputs.map((port) => (
            <Handle
              key={port.id}
              type="source"
              id={outHandle(port.id)}
              position={Position.Right}
              className="!opacity-0"
              style={{ top: -18 }}
            />
          ))}
          {spec.flow_out.map((name) => (
            <Handle
              key={name}
              type="source"
              id={flowOutHandle(name)}
              position={Position.Right}
              className="!opacity-0"
              style={{ top: -18 }}
            />
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-1.5 py-2">
          {(spec.inputs.length > 0 || spec.outputs.length > 0) && (
            <div className="flex justify-between gap-2">
              <div className="flex min-w-0 flex-col gap-1">
                {spec.inputs.map((port) => (
                  <div
                    key={port.id}
                    className="relative flex h-5 items-center pl-3 pr-1"
                  >
                    <Handle
                      type="target"
                      id={inHandle(port.id)}
                      position={Position.Left}
                      className="!h-2.5 !w-2.5 !border-2 !border-zinc-900"
                      style={{
                        background: portColorOf(portColors, port),
                        left: -5,
                      }}
                    />
                    <span className="flow-detail truncate text-[11px] text-zinc-400">
                      {port.label}
                      {port.multi ? "+" : ""}
                    </span>
                  </div>
                ))}
              </div>
              <div className="flex min-w-0 flex-col items-end gap-1">
                {spec.outputs.map((port) => (
                  <div
                    key={port.id}
                    className="relative flex h-5 items-center pl-1 pr-3"
                  >
                    <span className="flow-detail truncate text-[11px] text-zinc-400">
                      {port.label}
                    </span>
                    <Handle
                      type="source"
                      id={outHandle(port.id)}
                      position={Position.Right}
                      className="!h-2.5 !w-2.5 !border-2 !border-zinc-900"
                      style={{
                        background: portColorOf(portColors, port),
                        right: -5,
                      }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
          {spec.flow_out.length > 0 && (
            <div className="flex flex-col items-end gap-1">
              {spec.flow_out.map((name, i) => {
                const choices =
                  node.type === "story.choice"
                    ? (node.params.choices as string[] | undefined)
                    : undefined;
                const label = choices?.[i] ?? name.replace(/_/g, " ");
                if (choices && i >= choices.length) return null;
                return (
                  <div
                    key={name}
                    className="relative flex h-5 max-w-full items-center pl-1 pr-3"
                  >
                    <span className="flow-detail truncate text-[11px] italic text-zinc-300">
                      {label} ›
                    </span>
                    <Handle
                      type="source"
                      id={flowOutHandle(name)}
                      position={Position.Right}
                      className="!h-3 !w-3 !rounded-sm !border-2 !border-white !bg-zinc-900"
                      style={{ right: -6 }}
                    />
                  </div>
                );
              })}
            </div>
          )}
          {summary.length > 0 && (
            <div className="flex flex-col gap-0.5 px-3">
              {summary.map((line) => (
                <p
                  key={line}
                  className="flow-detail line-clamp-2 text-[11px] leading-snug text-zinc-500"
                >
                  {line}
                </p>
              ))}
            </div>
          )}
          {node.type === COLUMN_TYPE && <ColumnSample node={node} />}
          {spec.type === "train.finetune" && (
            <TrainNodeBody node={node} runtime={runtime} />
          )}
          {spec.type !== "train.finetune" &&
            (runtime.outputs ||
              runtime.partial ||
              runtime.text ||
              runtime.preview) && (
            <div className="flow-detail px-3">
              <OutputPreview runtime={runtime} spec={spec} />
            </div>
          )}
          {runtime.status === "failed" && runtime.error && (
            <p className="mx-3 line-clamp-3 rounded bg-red-500/10 px-2 py-1 text-[11px] text-red-300">
              {runtime.error}
            </p>
          )}
        </div>
      )}
      {running && (
        <div className="absolute inset-x-0 bottom-0 h-1 overflow-hidden rounded-b-xl bg-white/5">
          <div
            className={`h-full bg-sky-400 transition-[width] duration-300 ${runtime.progress === null ? "w-1/3 animate-pulse" : ""}`}
            style={
              runtime.progress === null
                ? undefined
                : { width: `${Math.round(runtime.progress * 100)}%` }
            }
          />
        </div>
      )}
    </div>
  );
}

export const FlowNodeView = memo(FlowNodeViewImpl);
