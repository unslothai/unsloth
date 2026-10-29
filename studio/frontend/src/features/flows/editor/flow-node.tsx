// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import {
  type ControlPosition,
  NodeResizeControl,
  type NodeProps,
  type OnResizeEnd,
  ResizeControlVariant,
  useStore,
} from "@xyflow/react";
import { memo } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import {
  IDLE_RUNTIME,
  type NodeRuntime,
  useRuntimeStore,
} from "../stores/runtime-store";
import type { FlowNode, NodeSpec, PortSpec } from "../types";
import {
  FLOW_IN_HANDLE,
  flowOutHandle,
  inHandle,
  outHandle,
} from "./connection-rules";
import { CARD_MIN_WIDTH, cardMinHeight } from "./card-size";
import type { RfNode } from "./derive-rf";
import { nodeIcon } from "./icons";
import { MediaThumb, isMediaValue } from "./media";
import {
  CollapsedPorts,
  type FieldRow,
  FlowInHandle,
  NodeCard,
  NodeCardHeader,
  NodeDisclosure,
  NodeError,
  NodeFieldRows,
  NodePorts,
  NodeStatusBadges,
  type PortRowSpec,
} from "./node-card";
import { NODE_LOOKS } from "./node-looks";
import { CATEGORY_TONES } from "./node-tones";
import { StructuredBadge, isStructuredValue } from "./value-viewers";

/** Below this zoom the text is unreadable anyway; hiding it (layout kept, so handles stay put)
 *  is what keeps panning a thousand-node graph smooth. */
const LOW_DETAIL_ZOOM = 0.45;
const lowDetailSelector = (s: { transform: [number, number, number] }) =>
  s.transform[2] < LOW_DETAIL_ZOOM;

function portColorOf(
  colors: Record<string, string> | undefined,
  port: PortSpec,
): string {
  return colors?.[port.type] ?? "#cbd5e1";
}

function shortText(value: unknown): string {
  const text = String(value).trim();
  return text.length > 60 ? `${text.slice(0, 60)}…` : text;
}

/** Up to three key settings as inline fields; long text goes behind "Show inputs". */
function nodeFields(
  node: FlowNode,
  spec: NodeSpec,
): { rows: FieldRow[]; long: { label: string; text: string }[] } {
  const rows: FieldRow[] = [];
  const long: { label: string; text: string }[] = [];
  for (const param of spec.params) {
    const value = node.params[param.id];
    if (value === null || value === undefined || value === "") continue;
    if (param.kind === "textarea") {
      const text = String(value).trim();
      if (text) long.push({ label: param.label, text });
      continue;
    }
    if (rows.length >= 3) continue;
    if (param.kind === "model") {
      const repo = (value as { repo?: string }).repo;
      if (repo) rows.push({ label: param.label, value: repo.split("/").pop() ?? repo });
      continue;
    }
    if (param.kind === "text" || param.kind === "select") {
      const text = shortText(value);
      if (text) rows.push({ label: param.label, value: text });
    }
  }
  return { rows, long };
}

function OutputPreview({
  runtime,
  spec,
  expanded = false,
}: { runtime: NodeRuntime; spec: NodeSpec; expanded?: boolean }) {
  if (
    runtime.preview &&
    (runtime.status === "running" || runtime.status === "loading_model")
  ) {
    return (
      <img
        src={runtime.preview}
        alt=""
        className="aspect-square w-full rounded-md object-cover group-data-[sized]/node:aspect-auto group-data-[sized]/node:h-full group-data-[sized]/node:object-contain"
        draggable={false}
      />
    );
  }
  if ((runtime.status === "running" || !runtime.outputs) && runtime.text) {
    return (
      <p className="line-clamp-4 whitespace-pre-wrap text-ui-11 leading-snug text-foreground group-data-[sized]/node:line-clamp-none">
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
        <span className="text-ui-10 font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-300">
          {at
            ? runtime.status === "running"
              ? `Partial: ${at.step} of ${at.total}`
              : `Partial (stopped at ${at.step} of ${at.total})`
            : "Partial"}
        </span>
        <OutputValues
          outputs={runtime.partial}
          spec={spec}
          expanded={expanded}
        />
      </div>
    );
  }
  const outputs = runtime.outputs;
  if (!outputs) return null;
  return <OutputValues outputs={outputs} spec={spec} expanded={expanded} />;
}

/** A list or object written out in full, for a card resized big enough to read it. */
export function expandedText(value: unknown): string {
  if (Array.isArray(value))
    return value
      .map((item) =>
        typeof item === "string" ? item : JSON.stringify(item, null, 2),
      )
      .join("\n");
  return JSON.stringify(value, null, 2);
}

function OutputValues({
  outputs,
  spec,
  expanded = false,
}: {
  outputs: Record<string, unknown>;
  spec: NodeSpec;
  expanded?: boolean;
}) {
  for (const port of spec.outputs) {
    const value = outputs[port.id];
    if (isMediaValue(value)) {
      return (
        <MediaThumb
          media={value}
          className="max-h-48 w-full rounded-md object-cover group-data-[sized]/node:h-full group-data-[sized]/node:max-h-full group-data-[sized]/node:object-contain"
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
    if (expanded && typeof value === "object") {
      return (
        <pre
          className="nowheel min-h-0 flex-1 overflow-auto whitespace-pre-wrap break-words font-mono text-ui-11 leading-snug text-foreground"
          data-testid="flow-output-expanded"
        >
          {expandedText(value)}
        </pre>
      );
    }
    if (typeof value === "string") {
      return (
        <p className="line-clamp-4 whitespace-pre-wrap text-ui-11 leading-snug text-foreground group-data-[sized]/node:line-clamp-none">
          {value}
        </p>
      );
    }
    if (typeof value === "number" || typeof value === "boolean") {
      return (
        <Badge variant="secondary" className="text-ui-11 tabular-nums">
          {String(value)}
        </Badge>
      );
    }
    if (Array.isArray(value)) {
      return (
        <Badge variant="secondary" className="text-ui-11">
          {value.length} items
        </Badge>
      );
    }
    return (
      <Badge variant="secondary" className="max-w-full font-mono text-ui-11">
        {"{ "}
        {Object.keys(value as object)
          .slice(0, 4)
          .join(", ")}
        {" }"}
      </Badge>
    );
  }
  return null;
}

const SIDES: ControlPosition[] = ["top", "bottom", "left", "right"];
const CORNERS: ControlPosition[] = [
  "top-left",
  "top-right",
  "bottom-left",
  "bottom-right",
];
/** Invisible edge and corner handles; one undo step per resize, committed on release. */
function CardResizer({ id, minHeight }: { id: string; minHeight: number }) {
  const onResizeEnd: OnResizeEnd = (_, p) =>
    useDocStore.getState().updateNode(id, {
      pos: [Math.round(p.x), Math.round(p.y)],
      size: [Math.round(p.width), Math.round(p.height)],
    });
  const common = { minWidth: CARD_MIN_WIDTH, minHeight, onResizeEnd };
  return (
    <>
      {SIDES.map((position) => (
        <NodeResizeControl
          key={position}
          {...common}
          position={position}
          variant={ResizeControlVariant.Line}
          className={
            position === "top" || position === "bottom"
              ? "!h-2 !border-0"
              : "!w-2 !border-0"
          }
        />
      ))}
      {CORNERS.map((position) => (
        <NodeResizeControl
          key={position}
          {...common}
          position={position}
          className="!size-3 !border-transparent !bg-transparent"
        />
      ))}
    </>
  );
}

function FlowNodeViewImpl({ id }: NodeProps<RfNode>) {
  const node = useDocStore((s) => s.graph.nodes[id]);
  const spec = useCatalogStore((s) => (node ? s.specs[node.type] : undefined));
  const category = useCatalogStore((s) =>
    spec ? s.categories[spec.category] : undefined,
  );
  const portColors = useCatalogStore((s) => s.catalog?.port_colors);
  const runtime = useRuntimeStore((s) => s.nodes[id]) ?? IDLE_RUNTIME;
  const lowDetail = useStore(lowDetailSelector);
  if (!node) return null;
  if (!spec) {
    return (
      <NodeCard runtime={runtime} className="border-destructive/60 p-3">
        <p className="text-xs text-destructive">
          Unknown node type <code>{node.type}</code>
        </p>
      </NodeCard>
    );
  }
  const look = NODE_LOOKS[node.type] ?? {};
  const title = node.label || spec.title;
  const categoryTitle = category?.title ?? spec.category;
  const subtitle = look.subtitle
    ? look.subtitle(node, spec)
    : node.label && node.label !== spec.title
      ? `${categoryTitle} · ${spec.title}`
      : categoryTitle;
  const collapsed = !!node.collapsed;
  const fields = collapsed || look.Body ? null : nodeFields(node, spec);
  const choices =
    node.type === "story.choice"
      ? (node.params.choices as string[] | undefined)
      : undefined;
  const flowOuts: PortRowSpec[] = spec.flow_out
    .map((name, i) => ({
      handleId: flowOutHandle(name),
      label: `${choices?.[i] ?? name.replace(/_/g, " ")} ›`,
      labelClassName: "italic text-foreground",
      hidden: !!choices && i >= choices.length,
    }))
    .filter((port) => !port.hidden);
  const { HeaderExtra, Ports, Body } = look;

  return (
    <NodeCard
      runtime={runtime}
      disabled={node.disabled}
      lowDetail={lowDetail}
      sized={!!node.size && !collapsed}
      data-node-id={id}
    >
      {/* First, so the port handles paint above the side handles. */}
      {!collapsed && <CardResizer id={id} minHeight={cardMinHeight(spec)} />}
      <NodeCardHeader
        icon={look.icon?.(node) ?? nodeIcon(spec.icon)}
        tone={look.tone ? look.tone(node) : CATEGORY_TONES[spec.category]}
        accent={look.accent?.(node) ?? category?.color ?? "#64748b"}
        title={title}
        subtitle={subtitle}
        leading={spec.flow_in ? <FlowInHandle id={FLOW_IN_HANDLE} /> : null}
      >
        {HeaderExtra && <HeaderExtra node={node} />}
        <NodeStatusBadges runtime={runtime} gpu={spec.lane === "gpu"} />
      </NodeCardHeader>

      {collapsed ? (
        Ports ? (
          <Ports node={node} collapsed={true} />
        ) : (
          <CollapsedPorts
            inputs={spec.inputs.map((port) => inHandle(port.id))}
            outputs={[
              ...spec.outputs.map((port) => outHandle(port.id)),
              ...spec.flow_out.map(flowOutHandle),
            ]}
          />
        )
      ) : (
        <div className="flex flex-col gap-1.5 py-2 group-data-[sized]/node:min-h-0 group-data-[sized]/node:flex-1 group-data-[sized]/node:[overflow-y:clip]">
          {Ports ? (
            <Ports node={node} collapsed={false} />
          ) : (
            <NodePorts
              inputs={spec.inputs.map((port) => ({
                handleId: inHandle(port.id),
                label: `${port.label}${port.multi ? "+" : ""}`,
                color: portColorOf(portColors, port),
              }))}
              outputs={spec.outputs.map((port) => ({
                handleId: outHandle(port.id),
                label: port.label,
                color: portColorOf(portColors, port),
              }))}
            />
          )}
          {flowOuts.length > 0 && <NodePorts inputs={[]} outputs={flowOuts} />}
          {Body ? <Body node={node} /> : null}
          {fields && <NodeFieldRows rows={fields.rows} />}
          {fields && <NodeDisclosure sections={fields.long} />}
          {(runtime.outputs ||
            runtime.partial ||
            runtime.text ||
            runtime.preview) && (
            <div className="flow-detail px-3 group-data-[sized]/node:flex group-data-[sized]/node:min-h-0 group-data-[sized]/node:flex-1 group-data-[sized]/node:flex-col">
              <OutputPreview
                runtime={runtime}
                spec={spec}
                expanded={!!node.size}
              />
            </div>
          )}
          {runtime.status === "failed" && runtime.error && (
            <NodeError message={runtime.error} />
          )}
        </div>
      )}
    </NodeCard>
  );
}

export const FlowNodeView = memo(FlowNodeViewImpl);
