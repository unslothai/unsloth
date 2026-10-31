// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { CollapsibleSectionTriggerButton } from "@/features/recipe-studio/dialogs/shared/collapsible-section-trigger";
import { FieldLabel } from "@/features/recipe-studio/dialogs/shared/field-label";
import { HugeiconsIcon } from "@hugeicons/react";
import { Suspense, lazy, memo, useCallback, useState } from "react";
import { loopMembers } from "../app/app-form";
import { useCatalogStore } from "../stores/catalog-store";
import { cloneFragment, useDocStore } from "../stores/doc-store";
import { IDLE_RUNTIME, useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import { removeNodes } from "../recipe/recipe-actions";
import {
  COLUMN_TYPE,
  GENERATE_TYPE,
  columnName,
} from "../recipe/recipe-columns";
import { ColumnPanel, RecipePanel } from "../recipe/recipe-panels";
import { SetupInspector } from "../recipe/recipe-setup-inspector";
import type { NodeSpec, ParamSpec } from "../types";
import {
  Cancel01Icon,
  Copy01Icon,
  Delete02Icon,
  DiceIcon,
  PlayIcon,
} from "./icon-data";
import { nodeIcon } from "./icons";
import { NodeIconTile } from "./node-card";
import { NODE_LOOKS } from "./node-looks";
import { NodeHistory } from "./node-history";
import { rerollNode, runToNode } from "./node-actions";
import { OutputValue } from "./output-value";
import { ParamField } from "./param-fields";
import { CATEGORY_TONES } from "./node-tones";
import { TriggerPanel } from "./trigger-panel";

const ColumnForm = lazy(() =>
  import("../recipe/recipe-forms").then((m) => ({ default: m.ColumnForm })),
);

function InspectorHeading({ children }: { children: string }) {
  return (
    <h3 className="text-xs font-semibold uppercase text-muted-foreground">
      {children}
    </h3>
  );
}

function NodeOutputs({ nodeId, spec }: { nodeId: string; spec: NodeSpec }) {
  const runtime = useRuntimeStore((s) => s.nodes[nodeId]) ?? IDLE_RUNTIME;
  const hasOutputs = runtime.outputs && Object.keys(runtime.outputs).length > 0;
  if (!hasOutputs && !runtime.text && !runtime.error) return null;
  return (
    <section className="flex flex-col gap-2 border-border/60 border-t px-4 py-3">
      <InspectorHeading>Output</InspectorHeading>
      {runtime.error && (
        <p className="rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
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
              <span className="text-xs text-muted-foreground">{port.label}</span>
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
    <div className="grid gap-1.5">
      <FieldLabel label={param.label} />
      {bound ? (
        <p className="rounded-lg border border-dashed border-border/60 px-3 py-2 text-xs text-muted-foreground">
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
  const [advancedOpen, setAdvancedOpen] = useState(false);
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

  if (!nodeId || !node || !spec) return <SetupInspector />;
  const isColumn = node.type === COLUMN_TYPE;
  const look = NODE_LOOKS[node.type];

  return (
    <aside
      className="flex h-full w-85 shrink-0 flex-col border-border border-l bg-background text-foreground"
      data-testid="flows-inspector"
    >
      <header className="flex items-start gap-2.5 border-border/60 border-b px-4 py-3">
        <NodeIconTile
          icon={look?.icon?.(node) ?? nodeIcon(spec.icon)}
          tone={look?.tone ? look.tone(node) : CATEGORY_TONES[spec.category]}
          accent={color}
        />
        <div className="min-w-0 flex-1">
          <input
            aria-label="Node name"
            value={isColumn ? columnName(node) : (node.label ?? "")}
            readOnly={isColumn}
            title={
              isColumn
                ? "Rename it in Column name below; references follow"
                : undefined
            }
            placeholder={spec.title}
            onChange={(e) =>
              useDocStore
                .getState()
                .updateNode(node.id, { label: e.target.value || null })
            }
            className="-mx-1.5 w-[calc(100%+0.75rem)] rounded-md bg-transparent px-1.5 py-0.5 font-semibold text-sm outline-none placeholder:text-foreground hover:bg-muted focus:bg-muted"
          />
          <p className="mt-0.5 text-ui-11 leading-snug text-muted-foreground">
            {isColumn
              ? `${look?.subtitle?.(node, spec) ?? "Column"} column of a Data Recipe`
              : spec.description}
          </p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label="Close inspector"
          onClick={() => useUiStore.getState().openInspector(null)}
          className="text-muted-foreground"
        >
          <HugeiconsIcon icon={Cancel01Icon} size={15} />
        </Button>
      </header>
      <div className="flex flex-wrap gap-1.5 border-border/60 border-b px-4 py-2">
        {!spec.design && (
          <>
            <Button
              type="button"
              variant="dark"
              size="xs"
              onClick={() => void runToNode(node.id)}
              title="Run only what this node needs"
            >
              <HugeiconsIcon icon={PlayIcon} size={12} /> Run to here
            </Button>
            <Button
              type="button"
              variant="outline"
              size="xs"
              onClick={() => void rerollNode(node.id)}
              title="Run this node again, ignoring the cache (new seed when it has one)"
            >
              <HugeiconsIcon icon={DiceIcon} size={12} /> Re-roll
            </Button>
            {node.type !== GENERATE_TYPE && (
              <Button
                type="button"
                variant="outline"
                size="xs"
                onClick={() => {
                  const doc = useDocStore.getState();
                  const fragment = cloneFragment(doc.graph, [node.id], [40, 40]);
                  doc.insertFragment(fragment);
                  useUiStore.getState().setSelection(
                    fragment.nodes.map((n) => n.id),
                    [],
                  );
                }}
              >
                <HugeiconsIcon icon={Copy01Icon} size={12} /> Duplicate
              </Button>
            )}
          </>
        )}
        <Button
          type="button"
          variant="outline"
          size="xs"
          onClick={() =>
            useDocStore
              .getState()
              .updateNode(node.id, { disabled: !node.disabled })
          }
        >
          {node.disabled ? "Enable" : "Disable"}
        </Button>
        <Button
          type="button"
          variant="outline"
          size="xs"
          onClick={() =>
            useDocStore
              .getState()
              .updateNode(node.id, { collapsed: !node.collapsed })
          }
        >
          {node.collapsed ? "Expand" : "Collapse"}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label="Delete node"
          onClick={() => {
            removeNodes([node.id]);
            useUiStore.getState().openInspector(null);
          }}
          className="ml-auto text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
        >
          <HugeiconsIcon icon={Delete02Icon} size={14} />
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {isColumn && (
          <section className="flex flex-col gap-4 px-4 py-3">
            <Suspense fallback={null}>
              <ColumnForm key={node.id} node={node} />
            </Suspense>
            <ColumnPanel node={node} />
          </section>
        )}
        {!isColumn && (
          <section className="flex flex-col gap-4 px-4 py-3">
            {spec.params.length === 0 && (
              <p className="text-xs text-muted-foreground">
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
              <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
                <CollapsibleTrigger asChild={true}>
                  <CollapsibleSectionTriggerButton
                    label="Advanced"
                    open={advancedOpen}
                  />
                </CollapsibleTrigger>
                <CollapsibleContent className="mt-3 flex flex-col gap-4">
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
                </CollapsibleContent>
              </Collapsible>
            )}
          </section>
        )}
        {node.type === GENERATE_TYPE && <RecipePanel node={node} />}
        {spec.type.startsWith("trigger.") && (
          <TriggerPanel nodeId={node.id} kind={spec.type} />
        )}
        <NodeOutputs nodeId={node.id} spec={spec} />
        {viewed && (
          <section className="flex flex-col gap-2 border-border/60 border-t px-4 py-3">
            <InspectorHeading>Earlier output</InspectorHeading>
            {(spec.outputs.length > 0
              ? spec.outputs
              : // Output blocks declare no ports but store what they showed.
                Object.keys(viewed).map((id) => ({ id, label: id }))
            ).map((port) =>
              port.id in viewed ? (
                <div key={port.id} className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">
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
