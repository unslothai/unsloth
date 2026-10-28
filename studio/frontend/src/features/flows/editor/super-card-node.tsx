// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import { type NodeProps, NodeToolbar, Position, useStore } from "@xyflow/react";
import { memo, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import { IDLE_RUNTIME, useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import { GROUP_PREFIX, type RfEdge, type RfNode } from "./derive-rf";
import { Maximize01Icon, PackageIcon } from "./icon-data";
import {
  NodeCard,
  NodeCardHeader,
  NodePorts,
  NodeStatusBadges,
} from "./node-card";
import { makeSuperCard, setSuperCard } from "./super-card-actions";
import {
  type SuperCardData,
  applySuperCards,
  collapsedGroups,
  rollUpStatus,
  superCardTitle,
} from "./super-cards";

const lowDetailSelector = (s: { transform: [number, number, number] }) =>
  s.transform[2] < 0.45;

function SuperCardViewImpl({ data }: NodeProps<RfNode>) {
  const { groupId, memberIds, inputs, outputs } = data as SuperCardData;
  const group = useDocStore((s) => s.graph.groups[groupId]);
  const rollup = useRuntimeStore(
    useShallow((s) => rollUpStatus(memberIds.map((id) => s.nodes[id]?.status))),
  );
  const lowDetail = useStore(lowDetailSelector);
  if (!group) return null;
  const steps = memberIds.length;
  const progress =
    rollup.status === "running" && steps
      ? ` · ${rollup.done} of ${steps} done`
      : "";
  return (
    <NodeCard
      runtime={{ ...IDLE_RUNTIME, status: rollup.status, progress: steps ? rollup.done / steps : null }}
      lowDetail={lowDetail}
      className="w-72"
      data-super-card={groupId}
      onDoubleClick={(event) => {
        event.stopPropagation();
        setSuperCard(groupId, false);
      }}
    >
      <NodeCardHeader
        icon={PackageIcon}
        accent={group.color ?? "#8b5cf6"}
        title={superCardTitle(group)}
        subtitle={`${steps} ${steps === 1 ? "step" : "steps"}${progress}`}
      >
        <NodeStatusBadges runtime={{ ...IDLE_RUNTIME, status: rollup.status }} />
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
            setSuperCard(groupId, false);
          }}
        >
          <HugeiconsIcon icon={Maximize01Icon} size={12} />
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

/** "Make super card" over a multi-selection, or "Collapse" over one selected group. */
function SuperCardSelectionToolbarImpl({ readOnly }: { readOnly: boolean }) {
  const target = useUiStore(
    useShallow((s) => {
      if (readOnly) return [] as string[];
      const { graph, meta } = useDocStore.getState();
      const frames = s.selectedNodes.filter((id) => id.startsWith(GROUP_PREFIX));
      const blocks = s.selectedNodes.filter((id) => graph.nodes[id]);
      if (frames.length === 1 && !blocks.length) {
        const groupId = frames[0].slice(GROUP_PREFIX.length);
        return collapsedGroups(meta.editor, graph).has(groupId) ? [] : frames;
      }
      return !frames.length && blocks.length >= 2 ? blocks : [];
    }),
  );
  if (!target.length) return null;
  const isFrame = target[0].startsWith(GROUP_PREFIX);
  return (
    <NodeToolbar
      nodeId={target}
      isVisible={true}
      position={Position.Top}
      offset={8}
      className="nodrag nopan nowheel flex items-center gap-1 rounded-lg border border-border bg-popover p-1 shadow-[var(--popup-shadow)]"
    >
      <Button
        type="button"
        variant="ghost"
        size="xs"
        title="Fold these blocks into one card; open it again any time"
        data-testid="make-super-card"
        onClick={() => void makeSuperCard()}
      >
        <HugeiconsIcon icon={PackageIcon} size={12} />
        {isFrame ? "Collapse to super card" : "Make super card"}
      </Button>
    </NodeToolbar>
  );
}

export const SuperCardSelectionToolbar = memo(SuperCardSelectionToolbarImpl);

/** The canvas's nodes and edges with every collapsed group folded into its card. */
export function useSuperCardView(
  nodes: RfNode[],
  edges: RfEdge[],
): { nodes: RfNode[]; edges: RfEdge[] } {
  const graph = useDocStore((s) => s.graph);
  const editor = useDocStore((s) => s.meta.editor);
  const specs = useCatalogStore((s) => s.specs);
  const portColors = useCatalogStore((s) => s.catalog?.port_colors);
  const collapsed = useMemo(
    () => collapsedGroups(editor, graph),
    [editor, graph],
  );
  return useMemo(
    () => applySuperCards(nodes, edges, graph, collapsed, specs, portColors),
    [nodes, edges, graph, collapsed, specs, portColors],
  );
}
