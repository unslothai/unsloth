// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type NodeProps,
  NodeToolbar,
  Position,
  type ReactFlowState,
  ViewportPortal,
  useReactFlow,
  useStore,
} from "@xyflow/react";
import { memo, useCallback, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import { IDLE_RUNTIME, useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import type { RfEdge, RfNode } from "./derive-rf";
import {
  ArrowExpand01Icon,
  ArrowShrink01Icon,
  Layers01Icon,
  UngroupLayersIcon,
} from "./icon-data";
import {
  NodeCard,
  NodeCardHeader,
  NodePorts,
  NodeStatusBadges,
} from "./node-card";
import {
  combineSelection,
  setSuperCardOpen,
  ungroupSuperCard,
} from "./super-card-actions";
import {
  SUPER_PREFIX,
  type SuperCardData,
  applySuperCards,
  combinableSelection,
  readSuperCards,
  rollUpStatus,
} from "./super-cards";

const FOCUS_MIN_ZOOM = 0.8;

const lowDetailSelector = (s: { transform: [number, number, number] }) =>
  s.transform[2] < 0.45;

function SuperCardViewImpl({ data }: NodeProps<RfNode>) {
  const { cardId, title, color, memberIds, inputs, outputs } =
    data as unknown as SuperCardData;
  const rollup = useRuntimeStore(
    useShallow((s) => rollUpStatus(memberIds.map((id) => s.nodes[id]?.status))),
  );
  const lowDetail = useStore(lowDetailSelector);
  const steps = memberIds.length;
  const progress =
    rollup.status === "running" && rollup.total > 1
      ? ` · ${rollup.done} of ${rollup.total} done`
      : "";
  return (
    <NodeCard
      runtime={{
        ...IDLE_RUNTIME,
        status: rollup.status,
        progress: rollup.total > 1 ? rollup.done / rollup.total : null,
      }}
      lowDetail={lowDetail}
      className="w-72"
      data-super-card={cardId}
      onDoubleClick={(event) => {
        event.stopPropagation();
        setSuperCardOpen(cardId, true);
      }}
    >
      <NodeCardHeader
        icon={Layers01Icon}
        accent={color ?? "#8b5cf6"}
        title={title}
        subtitle={`${steps} ${steps === 1 ? "block" : "blocks"}${progress}`}
      >
        <NodeStatusBadges
          runtime={{ ...IDLE_RUNTIME, status: rollup.status }}
        />
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          className="nodrag text-muted-foreground"
          title="Open (double-click)"
          aria-label="Open super card"
          data-testid="super-card-open"
          onClick={(event) => {
            event.stopPropagation();
            setSuperCardOpen(cardId, true);
          }}
        >
          <HugeiconsIcon icon={ArrowExpand01Icon} size={12} />
        </Button>
      </NodeCardHeader>
      <div className="py-2">
        {inputs.length || outputs.length ? (
          <NodePorts inputs={inputs} outputs={outputs} />
        ) : (
          <p className="flow-detail px-3 text-ui-11 text-muted-foreground">
            Nothing wired in or out
          </p>
        )}
      </div>
    </NodeCard>
  );
}

export const SuperCardView = memo(SuperCardViewImpl);

type Rect = { x: number; y: number; w: number; h: number };

function useRects(nodeIds: string[]): Rect[] {
  const key = useStore(
    useCallback(
      (state: ReactFlowState) =>
        JSON.stringify(
          nodeIds.flatMap((id): Rect[] => {
            const node = state.nodeLookup.get(id);
            if (!node) return [];
            const { x, y } = node.internals.positionAbsolute;
            return [
              {
                x,
                y,
                w: node.measured.width ?? 0,
                h: node.measured.height ?? 0,
              },
            ];
          }),
        ),
      [nodeIds],
    ),
  );
  return useMemo(() => JSON.parse(key) as Rect[], [key]);
}

function Outline({ rect, pad }: { rect: Rect; pad: number }) {
  return (
    <div
      className="pointer-events-none absolute rounded-2xl border-2 border-dashed border-primary/60"
      style={{
        transform: `translate(${rect.x - pad}px, ${rect.y - pad}px)`,
        width: rect.w + pad * 2,
        height: rect.h + pad * 2,
      }}
    />
  );
}

/** Two or more selected blocks get the combine button above them and a dashed outline each,
 *  the same affordance as Recipe Studio's super cards. */
function SuperCardCombineToolbarImpl({ readOnly }: { readOnly: boolean }) {
  const nodeIds = useUiStore(
    useShallow((s) => {
      if (readOnly) return [] as string[];
      const { graph, meta } = useDocStore.getState();
      return combinableSelection(
        s.selectedNodes,
        graph,
        readSuperCards(meta.editor, graph),
      );
    }),
  );
  const rects = useRects(nodeIds);
  const { fitView, getEdges } = useReactFlow();
  if (nodeIds.length < 2) return null;
  const onCombine = () => {
    const id = combineSelection();
    if (!id) return;
    const cardRfId = SUPER_PREFIX + id;
    // Frame the new card and what it wires to, never below a readable zoom.
    window.setTimeout(() => {
      const ids = new Set([cardRfId]);
      for (const edge of getEdges()) {
        if (edge.source === cardRfId) ids.add(edge.target);
        if (edge.target === cardRfId) ids.add(edge.source);
      }
      void fitView({
        nodes: [...ids].map((nodeId) => ({ id: nodeId })),
        minZoom: FOCUS_MIN_ZOOM,
        maxZoom: 1.1,
        padding: 0.25,
        duration: 300,
      });
    }, 60);
  };
  return (
    <>
      <ViewportPortal>
        {rects.map((rect, i) => (
          <Outline key={nodeIds[i]} rect={rect} pad={10} />
        ))}
      </ViewportPortal>
      <NodeToolbar
        nodeId={nodeIds}
        isVisible={true}
        position={Position.Top}
        offset={14}
        className="nodrag nopan"
      >
        <Button
          type="button"
          size="sm"
          className="corner-squircle shadow-md"
          data-testid="combine-super-card"
          onClick={onCombine}
        >
          <HugeiconsIcon icon={Layers01Icon} className="size-4" />
          Combine {nodeIds.length} blocks into a super card
        </Button>
      </NodeToolbar>
    </>
  );
}

export const SuperCardCombineToolbar = memo(SuperCardCombineToolbarImpl);

function OpenSuperCard({
  id,
  title,
  members,
  readOnly,
}: {
  id: string;
  title: string;
  members: string[];
  readOnly: boolean;
}) {
  const rects = useRects(members);
  if (!rects.length) return null;
  const x = Math.min(...rects.map((r) => r.x));
  const y = Math.min(...rects.map((r) => r.y));
  const box = {
    x,
    y,
    w: Math.max(...rects.map((r) => r.x + r.w)) - x,
    h: Math.max(...rects.map((r) => r.y + r.h)) - y,
  };
  return (
    <>
      <ViewportPortal>
        <Outline rect={box} pad={16} />
      </ViewportPortal>
      <NodeToolbar
        nodeId={members}
        isVisible={true}
        position={Position.Top}
        offset={24}
        className="nodrag nopan flex items-center gap-1 rounded-lg border border-border bg-popover p-1 pl-2 shadow-[var(--popup-shadow)]"
        data-open-super-card={id}
      >
        <HugeiconsIcon icon={Layers01Icon} size={14} className="text-primary" />
        <span className="max-w-48 truncate text-ui-12 font-medium">
          {title}
        </span>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          data-testid="super-card-collapse"
          onClick={() => setSuperCardOpen(id, false)}
        >
          <HugeiconsIcon icon={ArrowShrink01Icon} size={12} />
          Collapse
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          disabled={readOnly}
          data-testid="super-card-ungroup"
          onClick={() => ungroupSuperCard(id)}
        >
          <HugeiconsIcon icon={UngroupLayersIcon} size={12} />
          Ungroup
        </Button>
      </NodeToolbar>
    </>
  );
}

/** Open cards draw a dashed outline round their blocks, with Collapse and Ungroup above it. */
function OpenSuperCardsImpl({ readOnly }: { readOnly: boolean }) {
  const graph = useDocStore((s) => s.graph);
  const editor = useDocStore((s) => s.meta.editor);
  const open = useMemo(
    () => readSuperCards(editor, graph).filter((c) => c.open),
    [editor, graph],
  );
  return open.map((card) => (
    <OpenSuperCard
      key={card.id}
      id={card.id}
      title={card.title}
      members={card.members}
      readOnly={readOnly}
    />
  ));
}

export const OpenSuperCards = memo(OpenSuperCardsImpl);

/** The canvas's nodes and edges with every closed super card folded in. */
export function useSuperCardView(
  nodes: RfNode[],
  edges: RfEdge[],
  selected: ReadonlySet<string>,
): { nodes: RfNode[]; edges: RfEdge[] } {
  const graph = useDocStore((s) => s.graph);
  const editor = useDocStore((s) => s.meta.editor);
  const specs = useCatalogStore((s) => s.specs);
  const portColors = useCatalogStore((s) => s.catalog?.port_colors);
  const cards = useMemo(() => readSuperCards(editor, graph), [editor, graph]);
  return useMemo(
    () =>
      applySuperCards(nodes, edges, graph, cards, specs, portColors, selected),
    [nodes, edges, graph, cards, specs, portColors, selected],
  );
}
