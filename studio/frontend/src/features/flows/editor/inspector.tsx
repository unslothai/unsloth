// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useCallback } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { cloneFragment, useDocStore } from "../stores/doc-store";
import { IDLE_RUNTIME, useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import type { NodeSpec } from "../types";
import {
  Cancel01Icon,
  Copy01Icon,
  Delete02Icon,
  DiceIcon,
  PlayIcon,
} from "./icon-data";
import { nodeIcon } from "./icons";
import { MediaThumb, isMediaValue } from "./media";
import { ParamField } from "./param-fields";
import { runFlow } from "./run-controller";

function OutputValue({ value }: { value: unknown }) {
  if (value === null || value === undefined)
    return <span className="text-zinc-600">—</span>;
  if (isMediaValue(value)) {
    return (
      <MediaThumb media={value} className="w-full rounded-md object-contain" />
    );
  }
  if (Array.isArray(value) && value.length > 0 && value.every(isMediaValue)) {
    return (
      <div className="grid grid-cols-2 gap-1.5">
        {value.slice(0, 24).map((item, i) => (
          <MediaThumb
            key={`${(item as { ref: string }).ref}-${i}`}
            media={item}
            className="w-full rounded object-cover"
          />
        ))}
      </div>
    );
  }
  if (typeof value === "string") {
    return (
      <p className="max-h-72 overflow-y-auto whitespace-pre-wrap text-[12.5px] leading-relaxed text-zinc-200">
        {value}
      </p>
    );
  }
  return (
    <pre className="max-h-72 overflow-auto rounded-md bg-black/40 p-2 font-mono text-[11px] leading-snug text-zinc-300">
      {JSON.stringify(value, null, 2).slice(0, 20_000)}
    </pre>
  );
}

function NodeOutputs({ nodeId, spec }: { nodeId: string; spec: NodeSpec }) {
  const runtime = useRuntimeStore((s) => s.nodes[nodeId]) ?? IDLE_RUNTIME;
  const hasOutputs = runtime.outputs && Object.keys(runtime.outputs).length > 0;
  if (!hasOutputs && !runtime.text && !runtime.error) return null;
  return (
    <section className="flex flex-col gap-2 border-white/10 border-t px-4 py-3">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
        Output
      </h3>
      {runtime.error && (
        <p className="rounded bg-red-500/10 px-2 py-1.5 text-[12px] text-red-300">
          {runtime.error}
        </p>
      )}
      {!hasOutputs && runtime.text && <OutputValue value={runtime.text} />}
      {hasOutputs &&
        spec.outputs.map((port) =>
          runtime.outputs && port.id in runtime.outputs ? (
            <div key={port.id} className="flex flex-col gap-1">
              <span className="text-[11px] text-zinc-500">{port.label}</span>
              <OutputValue value={runtime.outputs[port.id]} />
            </div>
          ) : null,
        )}
    </section>
  );
}

function InspectorImpl() {
  const nodeId = useUiStore((s) => s.inspectorNodeId);
  const node = useDocStore((s) => (nodeId ? s.graph.nodes[nodeId] : undefined));
  const spec = useCatalogStore((s) => (node ? s.specs[node.type] : undefined));
  const color =
    useCatalogStore((s) =>
      spec ? s.categories[spec.category]?.color : undefined,
    ) ?? "#64748b";

  const setParam = useCallback(
    (paramId: string, value: unknown) => {
      if (!nodeId) return;
      useDocStore
        .getState()
        .updateParams(
          nodeId,
          { [paramId]: value },
          `param:${nodeId}:${paramId}`,
        );
    },
    [nodeId],
  );

  if (!nodeId || !node || !spec) return null;

  return (
    <aside
      className="flex h-full w-85 shrink-0 flex-col border-white/10 border-l bg-zinc-950/95 text-zinc-100"
      data-testid="flows-inspector"
    >
      <header
        className="flex items-start gap-2 border-white/10 border-b px-4 py-3"
        style={{ borderTop: `2px solid ${color}` }}
      >
        <HugeiconsIcon
          icon={nodeIcon(spec.icon)}
          size={18}
          strokeWidth={1.8}
          style={{ color }}
          className="mt-1"
        />
        <div className="min-w-0 flex-1">
          <input
            value={node.label ?? ""}
            placeholder={spec.title}
            onChange={(e) =>
              useDocStore
                .getState()
                .updateNode(node.id, { label: e.target.value || null })
            }
            className="w-full bg-transparent font-semibold text-[15px] outline-none placeholder:text-zinc-100"
          />
          <p className="mt-0.5 text-[11.5px] leading-snug text-zinc-500">
            {spec.description}
          </p>
        </div>
        <button
          type="button"
          aria-label="Close inspector"
          onClick={() => useUiStore.getState().openInspector(null)}
          className="rounded p-1 text-zinc-500 hover:bg-white/5 hover:text-zinc-200"
        >
          <HugeiconsIcon icon={Cancel01Icon} size={15} />
        </button>
      </header>
      <div className="flex flex-wrap gap-1.5 border-white/10 border-b px-4 py-2">
        <button
          type="button"
          onClick={() => void runFlow({ targetNodes: [node.id] })}
          className="flex items-center gap-1 rounded-md bg-sky-600/80 px-2 py-1 text-[12px] hover:bg-sky-500"
          title="Run only what this node needs"
        >
          <HugeiconsIcon icon={PlayIcon} size={12} /> Run to here
        </button>
        <button
          type="button"
          onClick={() => {
            const hasSeed = spec.params.some(
              (p) => p.id === "seed" && p.kind === "seed",
            );
            if (hasSeed) setParam("seed", Math.floor(Math.random() * 2 ** 31));
            void runFlow({ targetNodes: [node.id], forceNodes: [node.id] });
          }}
          className="flex items-center gap-1 rounded-md border border-white/10 px-2 py-1 text-[12px] hover:bg-white/5"
          title="Run this node again, ignoring the cache (new seed when it has one)"
        >
          <HugeiconsIcon icon={DiceIcon} size={12} /> Re-roll
        </button>
        <button
          type="button"
          onClick={() => {
            const doc = useDocStore.getState();
            const fragment = cloneFragment(doc.graph, [node.id], [40, 40]);
            doc.insertFragment(fragment);
            useUiStore.getState().setSelection(
              fragment.nodes.map((n) => n.id),
              [],
            );
          }}
          className="flex items-center gap-1 rounded-md border border-white/10 px-2 py-1 text-[12px] hover:bg-white/5"
        >
          <HugeiconsIcon icon={Copy01Icon} size={12} /> Duplicate
        </button>
        <button
          type="button"
          onClick={() =>
            useDocStore
              .getState()
              .updateNode(node.id, { disabled: !node.disabled })
          }
          className="rounded-md border border-white/10 px-2 py-1 text-[12px] hover:bg-white/5"
        >
          {node.disabled ? "Enable" : "Disable"}
        </button>
        <button
          type="button"
          onClick={() =>
            useDocStore
              .getState()
              .updateNode(node.id, { collapsed: !node.collapsed })
          }
          className="rounded-md border border-white/10 px-2 py-1 text-[12px] hover:bg-white/5"
        >
          {node.collapsed ? "Expand" : "Collapse"}
        </button>
        <button
          type="button"
          aria-label="Delete node"
          onClick={() => {
            useDocStore.getState().removeElements({ nodes: [node.id] });
            useUiStore.getState().openInspector(null);
          }}
          className="ml-auto rounded-md p-1 text-zinc-500 hover:bg-red-500/10 hover:text-red-300"
        >
          <HugeiconsIcon icon={Delete02Icon} size={14} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        <section className="flex flex-col gap-3 px-4 py-3">
          {spec.params.length === 0 && (
            <p className="text-[12px] text-zinc-500">
              This node has no settings.
            </p>
          )}
          {spec.params.map((param) => (
            <div key={param.id} className="flex flex-col gap-1">
              <span className="text-[11.5px] font-medium text-zinc-400">
                {param.label}
              </span>
              <ParamField
                spec={param}
                value={node.params[param.id]}
                onChange={(value) => setParam(param.id, value)}
              />
            </div>
          ))}
        </section>
        <NodeOutputs nodeId={node.id} spec={spec} />
      </div>
    </aside>
  );
}

export const Inspector = memo(InspectorImpl);
