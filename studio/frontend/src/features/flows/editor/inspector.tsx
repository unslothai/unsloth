// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useCallback, useState } from "react";
import { loopMembers } from "../app/app-form";
import { useCatalogStore } from "../stores/catalog-store";
import { cloneFragment, useDocStore } from "../stores/doc-store";
import { IDLE_RUNTIME, useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import { commitRecipe, removeNodes } from "../recipe/recipe-actions";
import {
  COLUMN_TYPE,
  GENERATE_TYPE,
  columnName,
} from "../recipe/recipe-columns";
import { ColumnPanel, RecipePanel } from "../recipe/recipe-panels";
import type { NodeSpec, ParamSpec } from "../types";
import {
  Cancel01Icon,
  Copy01Icon,
  Delete02Icon,
  DiceIcon,
  PlayIcon,
} from "./icon-data";
import { nodeIcon } from "./icons";
import { NodeHistory } from "./node-history";
import { rerollNode, runToNode } from "./node-actions";
import { OutputValue } from "./output-value";
import { ParamField } from "./param-fields";
import { TriggerPanel } from "./trigger-panel";


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
        spec.outputs.length === 0 &&
        Object.entries(runtime.outputs ?? {}).map(([key, value]) => (
          <OutputValue key={key} value={value} />
        ))}
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

function ParamRow({
  param,
  bound,
  value,
  onChange,
}: {
  param: ParamSpec;
  bound: boolean;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11.5px] font-medium text-zinc-400">
        {param.label}
      </span>
      {bound ? (
        <p className="rounded-md border border-dashed border-white/15 px-2 py-1.5 text-[12px] text-zinc-400">
          Set by its connected input
        </p>
      ) : (
        <ParamField spec={param} value={value} onChange={onChange} />
      )}
    </div>
  );
}

function InspectorImpl() {
  const nodeId = useUiStore((s) => s.inspectorNodeId);
  const node = useDocStore((s) => (nodeId ? s.graph.nodes[nodeId] : undefined));
  const inLoop = useDocStore(
    (s) => !!nodeId && loopMembers(Object.values(s.graph.nodes)).has(nodeId),
  );
  const spec = useCatalogStore((s) => (node ? s.specs[node.type] : undefined));
  const [viewed, setViewed] = useState<Record<string, unknown> | null>(null);
  const color =
    useCatalogStore((s) =>
      spec ? s.categories[spec.category]?.color : undefined,
    ) ?? "#64748b";
  const bound = useDocStore((s) => {
    if (!nodeId) return "";
    const ids: string[] = [];
    for (const edge of Object.values(s.graph.edges))
      if (edge.target === nodeId && edge.target_port.startsWith("param:"))
        ids.push(edge.target_port.slice("param:".length));
    return ids.sort().join("\n");
  });
  const boundParams = new Set(bound ? bound.split("\n") : []);

  const setParam = useCallback(
    (paramId: string, value: unknown) => {
      if (!nodeId) return;
      const current = useDocStore.getState().graph.nodes[nodeId];
      if (current?.type === COLUMN_TYPE && current.group) {
        // Raw column JSON: redraw the reference wires it implies.
        commitRecipe(
          current.group,
          [{ ...current, params: { ...current.params, [paramId]: value } }],
          null,
          `param:${nodeId}:${paramId}`,
        );
        return;
      }
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
  const isColumn = node.type === COLUMN_TYPE;

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
            value={isColumn ? columnName(node) : (node.label ?? "")}
            readOnly={isColumn}
            title={
              isColumn
                ? "Rename a column in its editor; references follow"
                : undefined
            }
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
        {!spec.design && (
          <>
            <button
              type="button"
              onClick={() => void runToNode(node.id)}
              className="flex items-center gap-1 rounded-md bg-sky-600/80 px-2 py-1 text-[12px] hover:bg-sky-500"
              title="Run only what this node needs"
            >
              <HugeiconsIcon icon={PlayIcon} size={12} /> Run to here
            </button>
            <button
              type="button"
              onClick={() => void rerollNode(node.id)}
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
          </>
        )}
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
            removeNodes([node.id]);
            useUiStore.getState().openInspector(null);
          }}
          className="ml-auto rounded-md p-1 text-zinc-500 hover:bg-red-500/10 hover:text-red-300"
        >
          <HugeiconsIcon icon={Delete02Icon} size={14} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {isColumn && <ColumnPanel node={node} />}
        <section className="flex flex-col gap-3 px-4 py-3">
          {spec.params.length === 0 && (
            <p className="text-[12px] text-zinc-500">
              This node has no settings.
            </p>
          )}
          {spec.params
            .filter((param) => !param.advanced)
            .map((param) => (
              <ParamRow
                key={param.id}
                param={param}
                bound={param.bindable === true && boundParams.has(param.id)}
                value={node.params[param.id]}
                onChange={(value) => setParam(param.id, value)}
              />
            ))}
          {spec.params.some((param) => param.advanced) && (
            <details>
              <summary className="cursor-pointer select-none text-[11.5px] font-medium text-zinc-400 hover:text-zinc-200">
                Advanced
              </summary>
              <div className="mt-3 flex flex-col gap-3">
                {spec.params
                  .filter((param) => param.advanced)
                  .map((param) => (
                    <ParamRow
                      key={param.id}
                      param={param}
                      bound={param.bindable === true && boundParams.has(param.id)}
                      value={node.params[param.id]}
                      onChange={(value) => setParam(param.id, value)}
                    />
                  ))}
              </div>
            </details>
          )}
        </section>
        {node.type === GENERATE_TYPE && <RecipePanel node={node} />}
        {spec.type.startsWith("trigger.") && (
          <TriggerPanel nodeId={node.id} kind={spec.type} />
        )}
        <NodeOutputs nodeId={node.id} spec={spec} />
        {viewed && (
          <section className="flex flex-col gap-2 border-white/10 border-t px-4 py-3">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
              Earlier output
            </h3>
            {(spec.outputs.length > 0
              ? spec.outputs
              : // Output blocks declare no ports but store what they showed.
                Object.keys(viewed).map((id) => ({ id, label: id }))
            ).map((port) =>
              port.id in viewed ? (
                <div key={port.id} className="flex flex-col gap-1">
                  <span className="text-[11px] text-zinc-500">
                    {port.label}
                  </span>
                  <OutputValue value={viewed[port.id]} />
                </div>
              ) : null,
            )}
          </section>
        )}
        <NodeHistory nodeId={node.id} onView={setViewed} canPin={!inLoop} />
      </div>
    </aside>
  );
}

export const Inspector = memo(InspectorImpl);
